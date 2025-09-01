#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL pipeline – V3.9+
(two-stage flavor: market + risk mode)
(no-skip-on-unknown + priors + fair-cost + progressive streak bonus + time-sorted + global streak
 + allow-duplicates-across-coupons + selective-threshold + policy-margin gate + curriculum gates + day-context)

Zmiany vs V3.9:
 • Akcja ma 4 wymiary: (market_act, skip_flag, close_flag, risk_flag[0=pewniak,1=value])
 • Progi selekcji zależne od trybu (pewniak/value) + dodatkowa bramka na margin polityki
 • Różnicowanie nagród wg trybu (większa nagroda i kara dla „pewniaka”)
 • Curriculum: łagodniejsze progi i łagodniejsze zaostrzanie progów w kolejnych przebiegach
 • Delikatny aux-consistency bonus ze zgodnością z priors bramkowymi
 • Progresywny bonus za serię trafień (kupon + global)
 • NOWE: „wstępny skan dnia” – globalny kontekst z priors całego dnia dołączony do obserwacji (stały wektor)
 • Fix: bezpieczne tworzenie katalogów (exist_ok=True), pewne tworzenie katalogu logów
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import Env, spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DODATKOWE IMPORTY DLA ROBUST LOADERA ---
import zipfile, io
import cloudpickle as cp

# ==============================
#  Auto-stop / cele (zapasowe)
# ==============================
TARGET_DAY_SCORE = 95.0
TARGET_HIT_PCT   = 98.0
MAX_PASSES       = 10
PATIENCE         = 3

# ==============================
#  Weekend windows
# ==============================
def _infer_local_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if "start_time" not in df.columns or df["start_time"].isna().all():
        return None
    ts = pd.to_datetime(df["start_time"], errors="coerce")
    if ts.notna().any():
        try:
            sample = ts.dropna().iloc[0]
            has_tz = getattr(sample, "tzinfo", None) is not None
        except Exception:
            has_tz = False
        if not has_tz:
            ts = ts.dt.tz_localize("UTC")
        local = ts.dt.tz_convert("Europe/Warsaw")
        return pd.Timestamp(local.dt.date.mode().iat[0])
    return None

def _index_batches_by_date(batches: list[pd.DataFrame]) -> dict[pd.Timestamp, list[int]]:
    by_date: dict[pd.Timestamp, list[int]] = {}
    for i, df in enumerate(batches):
        d = _infer_local_date(df)
        if d is None:
            continue
        d = d.normalize()
        by_date.setdefault(d, []).append(i)
    return by_date

def build_weekend_windows(
    batches: list[pd.DataFrame],
) -> list[tuple[str, list[pd.DataFrame], tuple[pd.Timestamp, pd.Timestamp]]]:
    by_date = _index_batches_by_date(batches)
    if not by_date:
        return []
    all_dates = sorted(by_date.keys())
    date_set = set(all_dates)
    windows: list[tuple[str, list[pd.DataFrame], tuple[pd.Timestamp, pd.Timestamp]]] = []
    for d in all_dates:
        if d.weekday() == 4:  # Fri
            d_next = d + pd.Timedelta(days=1)
            if d_next in date_set:
                idxs = by_date.get(d, []) + by_date.get(d_next, [])
                windows.append(("fri_sat", [batches[i] for i in idxs], (d, d_next)))
        if d.weekday() == 5:  # Sat
            d_next = d + pd.Timedelta(days=1)
            if d_next in date_set:
                idxs = by_date.get(d, []) + by_date.get(d_next, [])
                windows.append(("sat_sun", [batches[i] for i in idxs], (d, d_next)))
    return windows

# ==============================
#  Schema safety
# ==============================
REQUIRED_COLS_TRAINING: List[str] = [
    "payload_winner_id","payload_goals_home","payload_win_or_draw",
    "payload_teams_home_league_cards_red__percentage","payload_under_over",
    "payload_percent_away","payload_percent_home","payload_advice",
    "payload_goals_away","payload_teams_home_league_cards_red__total",
    "payload_winner_name","payload_winner_comment","payload_percent_draw",
    "id_fp","home_team","away_team","score_fulltime_home","score_fulltime_away",
    "score_halftime_home","score_halftime_away","start_time",
]

# ==============================
#  PREPROCESSOR
# ==============================
PreprocType = Union[ColumnTransformer, Pipeline]

def _ohe_compatible(**kwargs):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)

def _numeric_coerce(df: pd.DataFrame, meta_cols: set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" and col not in meta_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    return df

def build_preprocessor(df_feat: pd.DataFrame) -> Pipeline:
    is_odds = re.compile(r"_odds$").search
    is_pct  = re.compile(r"(_chance$|_percentage$|percent)").search

    categorical = [c for c in df_feat.columns if df_feat[c].dtype == "object"]
    numeric_log = [c for c in df_feat.columns if is_odds(c)]
    numeric_min = [c for c in df_feat.columns if is_pct(c)]
    numeric_std = [c for c in df_feat.columns
                   if c not in categorical + numeric_log + numeric_min + ["start_time"]]

    coltf = ColumnTransformer(
        transformers=[
            ("std", StandardScaler(), numeric_std),
            ("log", Pipeline([("log", FunctionTransformer(np.log1p, validate=False)),
                              ("sc",  StandardScaler())]), numeric_log),
            ("min", MinMaxScaler(),  numeric_min),
            ("cat", _ohe_compatible(), categorical),
            ("start_time", "passthrough", ["start_time"] if "start_time" in df_feat.columns else []),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    preproc = Pipeline([
        ("coltf", coltf),
        ("final_scaler", StandardScaler(with_mean=False))
    ])
    return preproc

def _get_feature_names_in(preprocess: PreprocType):
    if isinstance(preprocess, Pipeline):
        ct = preprocess.named_steps.get("coltf", None)
        return getattr(ct, "feature_names_in_", None)
    return getattr(preprocess, "feature_names_in_", None)

def align_columns(df_feat: pd.DataFrame, preprocess: PreprocType) -> pd.DataFrame:
    expected = _get_feature_names_in(preprocess)
    if expected is None:
        print("⚠️  Nie mogę odczytać feature_names_in_ – przechodzę bez align.")
        return df_feat
    missing  = [c for c in expected if c not in df_feat.columns]
    extra    = [c for c in df_feat.columns if c not in expected]
    if missing:
        df_feat[missing] = np.nan
        print(f"⚠️  Dodano brakujące kolumny: {missing}")
    if extra:
        df_feat = df_feat.drop(columns=extra)
        print(f"⚠️  Usunięto nadmiarowe kolumny: {extra}")
    return df_feat[list(expected)].copy()

def add_missing_columns(df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS_TRAINING if c not in df.columns]
    if missing:
        df[missing] = np.nan
        if verbose:
            print(f"⚠️  Added {len(missing)} missing columns → {missing}")
    elif verbose:
        print("✅ No missing columns – schema already complete.")
    return df

def _smart_skip_mask(df_raw: pd.DataFrame) -> np.ndarray:
    n = len(df_raw)
    if n == 0:
        return np.zeros(0, dtype=bool)
    skip = np.zeros(n, dtype=bool)

    odds_cols = [c for c in df_raw.columns if c.endswith("_odds")]
    if odds_cols:
        odds = df_raw[odds_cols].apply(pd.to_numeric, errors="coerce")
        very_low = (odds.min(axis=1) < 1.05)
        very_high = (odds.max(axis=1) > 20.0)
        skip |= (very_low | very_high).fillna(False).to_numpy()

    pct_pattern = re.compile(r"(_chance$|_percentage$|percent)")
    pct_cols = [c for c in df_raw.columns if pct_pattern.search(c)]
    if pct_cols:
        pcts = df_raw[pct_cols].apply(pd.to_numeric, errors="coerce")
        max_pct = pcts.max(axis=1)
        low_signal = (max_pct < 40)
        ph = df_raw.get("payload_percent_home")
        pa = df_raw.get("payload_percent_away")
        pd_ = df_raw.get("payload_percent_draw")
        if ph is not None and pa is not None and pd_ is not None:
            try:
                ph = pd.to_numeric(ph, errors="coerce")
                pa = pd.to_numeric(pa, errors="coerce")
                pd_ = pd.to_numeric(pd_, errors="coerce")
                low_sep = (ph.sub(pa).abs() < 5)
                strong_draw = (pd_ > 25)
                skip |= (low_signal | (low_sep & strong_draw)).fillna(False).to_numpy()
            except Exception:
                skip |= low_signal.fillna(False).to_numpy()
        else:
            skip |= low_signal.fillna(False).to_numpy()
    return skip

def build_skip_mask(df_raw: pd.DataFrame, *, nan_frac_threshold: float = 0.8) -> np.ndarray:
    META_COLS = {"home_team", "away_team", "id_fp"}
    TARGETS   = {"score_fulltime_home", "score_fulltime_away", "score_halftime_home", "score_halftime_away"}
    DROP_COLS = {
        "fixture_id", "id_pred", "fetched_at",
        "payload_h2h", "payload_teams_away_logo", "payload_teams_home_logo",
        "payload_teams_away_league_lineups", "payload_teams_home_league_lineups",
        "payload_league_flag", "payload_league_logo",
    }
    cols = [c for c in df_raw.columns if c not in META_COLS | TARGETS | DROP_COLS]
    if not cols:
        return np.zeros(len(df_raw), dtype=bool)

    frac_nan = df_raw[cols].isna().mean(axis=1).to_numpy()
    if len(df_raw) < 200:
        nan_frac_threshold = min(0.85, nan_frac_threshold + 0.05)

    nan_skip = (frac_nan > float(nan_frac_threshold))
    smart_skip = _smart_skip_mask(df_raw)
    return (nan_skip | smart_skip).astype(bool)

def _convert_start_time_inplace(df_feat: pd.DataFrame) -> None:
    if "start_time" in df_feat.columns:
        dt = pd.to_datetime(df_feat["start_time"], errors="coerce")
        try:
            sample = dt.dropna().iloc[0]
            has_tz = getattr(sample, "tzinfo", None) is not None
        except Exception:
            has_tz = False
        if not has_tz:
            dt = dt.dt.tz_localize("UTC")
        epoch_ns = dt.astype("int64")
        epoch_sec = (epoch_ns // 1_000_000_000).astype("float64")
        df_feat["start_time"] = np.where(np.isfinite(epoch_sec), epoch_sec, 0.0).astype(np.float32)

def prepare_batch(
    df: pd.DataFrame,
    preprocess: PreprocType | None,
    *,
    fit_if_none: bool = False,
    save_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], PreprocType | None, np.ndarray]:
    META_COLS = {"home_team", "away_team", "id_fp", "score_halftime_home", "score_halftime_away"}

    df = add_missing_columns(df.copy())

    for col in ("score_fulltime_home", "score_fulltime_away", "score_halftime_home", "score_halftime_away"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _numeric_coerce(df, META_COLS)

    DROP_COLS = [
        "fixture_id", "id_pred", "fetched_at",
        "payload_h2h", "payload_teams_away_logo", "payload_teams_home_logo",
        "payload_teams_away_league_lineups", "payload_teams_home_league_lineups",
        "payload_league_flag", "payload_league_logo",
    ]
    df_feat = df.drop(columns=DROP_COLS + ["score_fulltime_home", "score_fulltime_away", "score_halftime_home", "score_halftime_away"], errors="ignore").copy()

    skip_mask = build_skip_mask(df_feat, nan_frac_threshold=0.8)

    _convert_start_time_inplace(df_feat)

    if preprocess is None and fit_if_none:
        preprocess = build_preprocessor(df_feat)
        X = preprocess.fit_transform(df_feat).astype(np.float32)
        if save_path:
            joblib.dump(preprocess, save_path)
            print(f"🖴  Zapisano preprocesor → {save_path}")
    elif preprocess is not None:
        df_feat_aligned = align_columns(df_feat, preprocess)
        X = preprocess.transform(df_feat_aligned).astype(np.float32)
    else:
        raise ValueError("Preprocessor is None and fit_if_none is False.")

    y = df[["score_fulltime_home", "score_fulltime_away"]].astype(np.float32).to_numpy()

    meta_cols = [
        c
        for c in (
            "id_fp",
            "home_team",
            "away_team",
            "start_time",
            "score_halftime_home",
            "score_halftime_away",
        )
        if c in df.columns
    ]
    meta = df[meta_cols].to_dict(orient="records")

    def _col_val(cname: str, i: int):
        if cname in df.columns:
            try:
                return float(pd.to_numeric(df[cname], errors="coerce").iloc[i])
            except Exception:
                return np.nan
        return np.nan

    for i in range(len(meta)):
        meta[i]["prior_ph"] = _col_val("payload_percent_home", i) / 100.0
        meta[i]["prior_pa"] = _col_val("payload_percent_away", i) / 100.0
        meta[i]["prior_pd"] = _col_val("payload_percent_draw", i) / 100.0
        meta[i]["prior_gh"] = _col_val("payload_goals_home", i)
        meta[i]["prior_ga"] = _col_val("payload_goals_away", i)

    return X, y, meta, preprocess, skip_mask

# ==============================
#  Wrapper: imputacja + maska
# ==============================
class ImputeNaNWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env, impute_value: float = 0.0):
        super().__init__(env)
        self.impute_value = impute_value

        orig_low: np.ndarray = self.observation_space.low
        orig_high: np.ndarray = self.observation_space.high

        mask_low = np.zeros_like(orig_low)
        mask_high = np.ones_like(orig_high)

        self.observation_space = spaces.Box(
            low=np.concatenate([orig_low, mask_low]),
            high=np.concatenate([orig_high, mask_high]),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        mask = np.isnan(obs).astype(np.float32)
        obs = np.nan_to_num(obs, nan=self.impute_value).astype(np.float32)
        out = np.concatenate([obs, mask])
        return out.astype(np.float32)

def cosine_schedule(start: float, end: float):
    def _sched(progress_remaining: float):
        ratio = 1.0 - progress_remaining
        cos_v = 0.5 * (1 + np.cos(np.pi * ratio))
        return end + (start - end) * cos_v
    return _sched

# ==============================
#  Callback: policy margin (top1-top2)
# ==============================
class PolicyMarginCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            env = self.model.get_env()
            obs = self.locals.get("obs", None)
            if obs is None:
                return True
            obs_t = torch.as_tensor(obs, device=self.model.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t)
                # MultiCategorical → dist.dists: [market, skip, close, risk]
                if hasattr(dist, "dists") and len(dist.dists) >= 1:
                    logits = dist.dists[0].logits  # [batch, 9] dla marketu
                    probs = F.softmax(logits, dim=-1)
                    top2 = torch.topk(probs, k=2, dim=-1).values[0]  # (2,)
                    margin = float(top2[0] - top2[1])
                    top1p = float(top2[0])
                    env.env_method("set_policy_metrics", pct=top1p, margin_pp=margin, top3=None)
        except Exception:
            pass
        return True

# ==============================
#  Środowisko
# ==============================
class StadiumMatchEnv(Env):
    """
    Akcja = (market_act, skip_flag, close_flag, risk_flag)
      • market_act ∈ {0..8}
      • skip_flag ∈ {0,1}
      • close_flag ∈ {0,1}
      • risk_flag ∈ {0=pewniak, 1=value}
    """
    metadata = {"render.modes": ["human"]}
    BET_NAMES = [
        "1/1",
        "1X",
        "1/2",
        "X/1",
        "X/X",
        "X/2",
        "2/1",
        "2/X",
        "2/2",
    ]

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: List[Dict[str, Any]],
        *,
        total_units: float = 20.0,
        min_bets_to_close: int = 25,
        max_bets_to_close: int = 35,
        coupon_price: float = 2.0,
        skip_mask: Optional[np.ndarray] = None,
        min_prior_for_bet: float = 0.65,       # legacy
        auto_skip_penalty_coef: float = 0.2,   # delikatna kara za auto-skip
        seed: int = 42,
    ):
        super().__init__()
        self.X, self.y, self.meta = X, y, meta

        # ---- Nagrody / balans – pro-accuracy ----
        self.base_per_correct = 10.0
        self.match_bonus_per_correct = 1.2
        self.perfect_bonus_per_cost = 6.0
        self.wrong_penalty = 12.0
        self.wide_weights = {}
        self.length_bonus_per_bet = 1.0
        self.step_hit_bonus = 0.6
        self.step_miss_pen  = 0.8

        self.skip_good_bonus   = 0.8
        self.skip_bad_penalty  = 0.6

        self.max_same_market_ratio = 0.55
        self.same_market_penalty   = 0.4
        self.diversity_step_bonus  = 0.8
        self.diversity_close_bonus = 0.5
        self.monotony_hard_penalty = 0.0
        self.market_specific_penalty = {}

        self.prior_topk = 2
        self.prior_threshold_low = 0.18
        self.invalid_prior_penalty = 0.5
        self.prior_align_bonus_known = 0.25
        self.prior_align_bonus_unknown = 0.35

        # Streak bonuses (kupon – progresywny)
        self.streak_step_bonus  = 0.30
        self.streak_close_bonus = 0.80

        # Globalny streak (po czasie – progresywny)
        self.global_streak_step_bonus = 0.25
        self._global_streak: int = 0
        self._global_longest_streak: int = 0

        # Selective prediction
        self.min_prior_for_bet = float(min_prior_for_bet)      # legacy
        self.auto_skip_penalty_coef = float(auto_skip_penalty_coef)

        # Trybowe progi (nadpisywane w curriculum; bazowo niższe)
        self.min_prior_sure  = 0.54
        self.min_prior_value = 0.22
        self.min_policy_margin = 0.12  # delikatniej dla „pewniaka”

        # Trybowe mnożniki nagród
        self.mode_hit_boost   = {0: 1.35, 1: 1.00}
        self.mode_miss_boost  = {0: 1.35, 1: 1.00}
        self.mode_div_bonus   = {0: 0.6,  1: 1.0}
        self.aux_consistency_coef = 0.15

        self.total_units = float(total_units)
        self.remaining_units = float(total_units)
        self.min_bets_to_close = int(min_bets_to_close)
        self.max_bets_to_close = int(max_bets_to_close)
        self.coupon_price = float(coupon_price)

        self.coupon: List[Tuple[int, int, Optional[bool], float, bool, int]] = []
        self.coupon_cost: float = 0.0
        self.bets_in_coupon = 0

        self.num_markets = 9
        self.market_counts = np.zeros(self.num_markets, dtype=np.int32)

        self._all_idxs: List[int] = []
        self._coupon_pool: List[int] = []
        self._coupon_pos: int = 0
        self.current_idx: int = -1

        self.skipped: List[Tuple[int, str, float]] = []
        self.used_match_ids: set[str] = set()
        self._coupon_taken_keys: set[str] = set()

        self._coupon_streak: int = 0
        self._coupon_longest_streak: int = 0
        self._coupon_series_active: bool = True

        if skip_mask is None:
            self.skip_mask = np.zeros(len(X), dtype=bool)
        else:
            self.skip_mask = np.asarray(skip_mask, dtype=bool)
            if len(self.skip_mask) != len(X):
                raise ValueError("skip_mask must have same length as X")

        n_features = X.shape[1]
        self.hist_dim = self.num_markets
        self.prior_dim = self.num_markets

        # --- KONTEKST DNIA (9 mean + 9 max + 1 count_norm) ---
        self.ctx_dim = self.num_markets * 2 + 1
        self._day_ctx = np.zeros(self.ctx_dim, dtype=np.float32)

        obs_low  = np.concatenate([
            np.full(n_features, -1e9, dtype=np.float32),
            np.zeros(self.hist_dim, dtype=np.float32),
            np.zeros(self.prior_dim, dtype=np.float32),
            np.zeros(self.ctx_dim, dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.full(n_features,  1e9, dtype=np.float32),
            np.ones(self.hist_dim, dtype=np.float32),
            np.ones(self.prior_dim, dtype=np.float32),
            np.ones(self.ctx_dim, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 4-wymiarowe MultiDiscrete
        self.action_space = spaces.MultiDiscrete([self.num_markets, 2, 2, 2])

        self.total_reward = 0.0
        self.total_max_points = 0.0
        self.global_bets = 0
        self.global_correct = 0

        self._last_coupon_total = 0
        self._last_coupon_correct = 0
        self._last_close_info: Dict[str, Any] = {}

        self._np_random, _ = gym.utils.seeding.np_random(seed)

        self._policy_pct: Optional[float] = None
        self._policy_margin_pp: Optional[float] = None
        self._policy_top3: Optional[list] = None

    # === KLUCZE MECZU / DUPLIKATY ===
    def _norm_id(self, v) -> Optional[str]:
        if v is None:
            return None
        try:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
        except Exception:
            pass
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        if re.fullmatch(r"\d+\.0", s):  # 123.0 -> 123
            s = s[:-2]
        return s

    def _match_keys(self, idx: int) -> list[str]:
        m = self.meta[idx] if 0 <= idx < len(self.meta) else {}
        home = str(m.get("home_team", "")).strip().lower()
        away = str(m.get("away_team", "")).strip().lower()

        keys: list[str] = []
        idn = self._norm_id(m.get("id_fp"))
        if idn:
            keys.append(f"id:{idn}")

        ts_key = ""
        try:
            raw = m.get("start_time", None)
            ts = pd.to_datetime(raw, errors="coerce")
            if pd.notna(ts):
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.tz_localize("UTC")
                ts_key = ts.tz_convert("Europe/Warsaw").strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        if ts_key:
            keys.append(f"ha:{home}|{away}|t:{ts_key}")
            keys.append(f"ha:{home}|{away}|d:{ts_key[:10]}")
        else:
            keys.append(f"ha:{home}|{away}")

        return keys

    def _sort_key(self, idx: int) -> tuple:
        m = self.meta[idx] if 0 <= idx < len(self.meta) else {}
        try:
            raw = m.get("start_time", None)
            ts = pd.to_datetime(raw, errors="coerce")
            if pd.isna(ts):
                ts_val = float("inf")
            else:
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.tz_localize("UTC")
                ts_val = ts.tz_convert("Europe/Warsaw").timestamp()
        except Exception:
            ts_val = float("inf")
        home = str(m.get("home_team", "")).strip().lower()
        away = str(m.get("away_team", "")).strip().lower()
        idn = self._norm_id(m.get("id_fp")) or f"row{idx}"
        return (ts_val, home, away, idn)

    def set_policy_metrics(self, pct: Optional[float] = None,
                           margin_pp: Optional[float] = None,
                           top3: Optional[list] = None) -> None:
        self._policy_pct = None if pct is None else float(pct)
        self._policy_margin_pp = None if margin_pp is None else float(margin_pp)
        self._policy_top3 = top3

    def _skip_match(self, idx: int, reason: str, delta: float = 0.0) -> None:
        self.skipped.append((idx, reason, float(delta)))

    @staticmethod
    def _sigmoid(x: float, k: float = 4.0) -> float:
        return float(1.0 / (1.0 + np.exp(-k * x)))

    def _market_priors_for_idx(self, idx: int) -> np.ndarray:
        """Szacuje prawdopodobieństwa dla 9 rynków HT/FT."""
        if not (0 <= idx < len(self.meta)):
            return np.zeros(self.num_markets, dtype=np.float32)

        m = self.meta[idx]
        ph = float(m.get("prior_ph", np.nan))
        pa = float(m.get("prior_pa", np.nan))
        pd_ = float(m.get("prior_pd", np.nan))
        gh = float(m.get("prior_gh", np.nan))
        ga = float(m.get("prior_ga", np.nan))

        if not np.isfinite(ph): ph = 1 / 3
        if not np.isfinite(pa): pa = 1 / 3
        if not np.isfinite(pd_): pd_ = 1 / 3
        if not np.isfinite(gh): gh = 1.1
        if not np.isfinite(ga): ga = 1.0

        # Przybliżenie rozkładu przerwowego na podstawie różnicy goli
        margin_ht = (gh - ga) * 0.6
        p_ht1 = self._sigmoid(margin_ht)
        p_ht2 = self._sigmoid(-margin_ht)
        p_htx = max(0.0, 1.0 - p_ht1 - p_ht2)

        priors = np.array([
            p_ht1 * ph,  # 1/1
            p_ht1 * pd_, # 1/X
            p_ht1 * pa,  # 1/2
            p_htx * ph,  # X/1
            p_htx * pd_, # X/X
            p_htx * pa,  # X/2
            p_ht2 * ph,  # 2/1
            p_ht2 * pd_, # 2/X
            p_ht2 * pa,  # 2/2
        ], dtype=np.float32)

        s = float(priors.sum())
        if s > 0:
            priors /= s
        return np.clip(priors, 0.0, 1.0)

    # === NOWE: KONTEKST DNIA ===
    def _compute_day_context(self, idxs: List[int]) -> np.ndarray:
        """
        Agreguje priory ze WSZYSTKICH meczów dnia i zwraca stały wektor:
        [priors_mean(9), priors_max(9), count_norm(1)] ∈ [0,1].
        """
        if not idxs:
            return np.zeros(self.ctx_dim, dtype=np.float32)

        priors_stack = []
        for i in idxs:
            pri = self._market_priors_for_idx(i)  # (9,)
            priors_stack.append(pri)
        P = np.stack(priors_stack, axis=0)  # [N, 9]

        priors_mean = np.clip(np.nanmean(P, axis=0), 0.0, 1.0)
        priors_max  = np.clip(np.nanmax(P,  axis=0), 0.0, 1.0)
        count_norm  = np.array([min(1.0, len(idxs) / 64.0)], dtype=np.float32)

        return np.concatenate([priors_mean.astype(np.float32),
                               priors_max.astype(np.float32),
                               count_norm], dtype=np.float32)

    def _inject_common_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        md = None
        if 0 <= self.current_idx < len(self.meta):
            try:
                raw_ts = self.meta[self.current_idx].get("start_time", None)
                if raw_ts is not None:
                    ts = pd.to_datetime(raw_ts, errors="coerce")
                    if pd.notna(ts):
                        has_tz = getattr(ts, "tzinfo", None) is not None
                        if not has_tz:
                            ts = ts.tz_localize("UTC")
                        md = ts.tz_convert("Europe/Warsaw").strftime("%Y-%m-%d")
            except Exception:
                md = None
        info["match_date"] = md
        info["n_features"] = int(self.X.shape[1]) if isinstance(self.X, np.ndarray) else None
        info["global_streak"] = int(self._global_streak)
        info["global_longest_streak"] = int(self._global_longest_streak)
        return info

    def _build_all_idxs(self):
        valid = [i for i in range(len(self.X)) if not bool(self.skip_mask[i])]
        valid.sort(key=self._sort_key)
        self._all_idxs = valid

    def _reset_coupon_pool(self):
        self._coupon_pool = list(self._all_idxs)
        self._coupon_pos = 0
        self.coupon.clear()
        self.coupon_cost = 0.0
        self.bets_in_coupon = 0
        self.market_counts[:] = 0
        self.skipped.clear()
        self._coupon_streak = 0
        self._coupon_longest_streak = 0
        self._coupon_series_active = True
        self._coupon_taken_keys.clear()

    def _draw_next_from_coupon(self) -> bool:
        while self._coupon_pos < len(self._coupon_pool):
            idx = int(self._coupon_pool[self._coupon_pos])
            self._coupon_pos += 1
            keys = self._match_keys(idx)
            if any(k in self._coupon_taken_keys for k in keys):
                continue
            self.current_idx = idx
            return True
        self.current_idx = -1
        return False

    def _has_current(self) -> bool:
        return 0 <= self.current_idx < len(self.X)

    def _hist_vec(self) -> np.ndarray:
        total = max(self.market_counts.sum(), 1)
        return (self.market_counts / total).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        if not self._has_current():
            return np.concatenate([
                np.zeros(self.X.shape[1], dtype=np.float32),
                np.zeros(self.hist_dim, dtype=np.float32),
                np.zeros(self.prior_dim, dtype=np.float32),
                self._day_ctx,
            ], dtype=np.float32)
        priors = self._market_priors_for_idx(self.current_idx)
        return np.concatenate([
            self.X[self.current_idx].astype(np.float32),
            self._hist_vec(),
            priors,
            self._day_ctx,
        ], dtype=np.float32)

    @staticmethod
    def _evaluate_prediction(h_ht: int, a_ht: int, h_ft: int, a_ft: int, action: int) -> bool:
        return [
            h_ht > a_ht and h_ft > a_ft,   # 0  1/1
            h_ht > a_ht and h_ft == a_ft,  # 1  1/X
            h_ht > a_ht and h_ft < a_ft,   # 2  1/2
            h_ht == a_ht and h_ft > a_ft,  # 3  X/1
            h_ht == a_ht and h_ft == a_ft, # 4  X/X
            h_ht == a_ht and h_ft < a_ft,  # 5  X/2
            h_ht < a_ht and h_ft > a_ft,   # 6  2/1
            h_ht < a_ht and h_ft == a_ft,  # 7  2/X
            h_ht < a_ht and h_ft < a_ft,   # 8  2/2
        ][action]

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        self.remaining_units = self.total_units
        self.total_reward = 0.0
        self.total_max_points = 0.0
        self.global_bets = 0
        self.global_correct = 0
        self._last_close_info = {}
        self._last_coupon_total = 0
        self._last_coupon_correct = 0

        self._global_streak = 0
        self._global_longest_streak = 0
        self.used_match_ids.clear()

        self._build_all_idxs()

        # KONTEKST DNIA – policz raz na bazie wszystkich widocznych indeksów
        try:
            self._day_ctx = self._compute_day_context(self._all_idxs)
        except Exception:
            self._day_ctx = np.zeros(self.ctx_dim, dtype=np.float32)

        self._reset_coupon_pool()

        if not self._draw_next_from_coupon():
            self.current_idx = -1

        return self._get_observation(), {}

    def _emergency_close(self) -> float:
        self._last_close_info = {}
        if self.bets_in_coupon <= 0 or self.remaining_units < self.coupon_price:
            return 0.0
        missing = max(0, self.min_bets_to_close - self.bets_in_coupon)
        penalty = float(missing) * 2.0

        before_reward = self.total_reward
        before_max = self.total_max_points
        before_hits = self.global_correct
        before_bets = self.global_bets

        r = self._close_coupon()

        self.total_reward -= penalty
        r -= penalty

        self._last_close_info = {
            "coupon_closed": True,
            "emergency_close": True,
            "coupon_reward": float(max(0.0, self.total_reward - before_reward)),
            "coupon_max": float(self.total_max_points - before_max),
            "coupon_hits": int(self.global_correct - before_hits),
            "coupon_bets": int(self.global_bets - before_bets),
            "penalty": float(penalty),
            "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
            "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
            "perfect_coupon": bool(
                getattr(self, "_last_coupon_total", 0) > 0 and
                getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
            ),
        }
        return r

    def _after_coupon_closed(self, reward_acc: float, info: Dict[str, Any]) -> tuple[float, Dict[str, Any], bool]:
        if self.remaining_units >= self.coupon_price and len(self._all_idxs) > 0:
            self._reset_coupon_pool()
            if not self._draw_next_from_coupon():
                return reward_acc, info, True
            return reward_acc, info, False
        else:
            return reward_acc, info, True

    def step(self, action):
        # 4 komponenty
        market_act, skip_flag, close_flag, risk_flag = map(int, action)
        is_sure = (risk_flag == 0)

        reward = 0.0
        terminated = False
        info: Dict[str, Any] = {}
        self._last_close_info = {}

        reward -= self.market_specific_penalty.get(market_act, 0.0)

        if not self._has_current():
            terminated = True
            if self.bets_in_coupon > 0 and self.remaining_units >= self.coupon_price:
                before_reward = self.total_reward
                before_max = self.total_max_points
                before_hits = self.global_correct
                before_bets = self.global_bets
                reward += self._close_coupon()
                info.update({
                    "coupon_closed": True,
                    "emergency_close": False,
                    "coupon_reward": float(self.total_reward - before_reward),
                    "coupon_max": float(self.total_max_points - before_max),
                    "coupon_hits": int(self.global_correct - before_hits),
                    "coupon_bets": int(self.global_bets - before_bets),
                    "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                    "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                    "perfect_coupon": bool(
                        getattr(self, "_last_coupon_total", 0) > 0 and
                        getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                    ),
                })
            info = self._inject_common_info(info)
            return self._get_observation(), reward, terminated, False, info

        idx = self.current_idx
        id_fp = self.meta[idx].get("id_fp", f"row{idx}")

        keys_now = self._match_keys(idx)
        if any(k in self._coupon_taken_keys for k in keys_now):
            self._skip_match(idx, "duplicate_in_coupon", 0.0)
            if not self._draw_next_from_coupon():
                if self.bets_in_coupon >= self.min_bets_to_close and self.remaining_units >= self.coupon_price:
                    before_reward = self.total_reward
                    before_max = self.total_max_points
                    before_hits = self.global_correct
                    before_bets = self.global_bets
                    reward += self._close_coupon()
                    info.update({
                        "coupon_closed": True,
                        "emergency_close": False,
                        "coupon_reward": float(self.total_reward - before_reward),
                        "coupon_max": float(self.total_max_points - before_max),
                        "coupon_hits": int(self.global_correct - before_hits),
                        "coupon_bets": int(self.global_bets - before_bets),
                        "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                        "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                        "perfect_coupon": bool(
                            getattr(self, "_last_coupon_total", 0) > 0 and
                            getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                        ),
                    })
                else:
                    reward += self._emergency_close()
                    info.update(self._last_close_info)
                reward, info, terminated = self._after_coupon_closed(reward, info)
            info = self._inject_common_info(info)
            return self._get_observation(), reward, terminated, False, info

        h_goals, a_goals = self.y[idx]
        h_ht = self.meta[idx].get("score_halftime_home")
        a_ht = self.meta[idx].get("score_halftime_away")
        is_unknown = (
            np.isnan(h_goals) or np.isnan(a_goals) or
            h_ht is None or a_ht is None or
            pd.isna(h_ht) or pd.isna(a_ht)
        )

        priors = self._market_priors_for_idx(idx)
        prior_score = float(priors[market_act])

        # kara za bardzo niski prior względem miękkiego progu
        if prior_score < self.prior_threshold_low:
            reward -= self.invalid_prior_penalty * (self.prior_threshold_low - prior_score + 1e-3)

        # --- SELECTIVE (twarde) bramki zależne od trybu + margin polityki (dla pewniaka)
        gate = self.min_prior_sure if is_sure else self.min_prior_value
        if skip_flag == 0 and prior_score < gate:
            delta_auto = -self.auto_skip_penalty_coef * (gate - prior_score)
            reward += delta_auto
            self._skip_match(idx, "auto_skip_low_prior_mode", delta_auto)
            if not self._draw_next_from_coupon():
                if self.bets_in_coupon >= self.min_bets_to_close and self.remaining_units >= self.coupon_price:
                    before_reward = self.total_reward
                    before_max = self.total_max_points
                    before_hits = self.global_correct
                    before_bets = self.global_bets
                    reward += self._close_coupon()
                    info.update({
                        "coupon_closed": True,
                        "emergency_close": False,
                        "coupon_reward": float(self.total_reward - before_reward),
                        "coupon_max": float(self.total_max_points - before_max),
                        "coupon_hits": int(self.global_correct - before_hits),
                        "coupon_bets": int(self.global_bets - before_bets),
                        "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                        "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                        "perfect_coupon": bool(
                            getattr(self, "_last_coupon_total", 0) > 0 and
                            getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                        ),
                    })
                else:
                    reward += self._emergency_close()
                    info.update(self._last_close_info)
                reward, info, terminated = self._after_coupon_closed(reward, info)
            info = self._inject_common_info(info)
            return self._get_observation(), reward, terminated, False, info

        if skip_flag == 0 and is_sure and (self._policy_margin_pp is not None):
            if self._policy_margin_pp < self.min_policy_margin:
                delta_auto = -0.1 * (self.min_policy_margin - self._policy_margin_pp)
                reward += delta_auto
                self._skip_match(idx, "auto_skip_low_policy_margin", delta_auto)
                if not self._draw_next_from_coupon():
                    if self.bets_in_coupon >= self.min_bets_to_close and self.remaining_units >= self.coupon_price:
                        before_reward = self.total_reward
                        before_max = self.total_max_points
                        before_hits = self.global_correct
                        before_bets = self.global_bets
                        reward += self._close_coupon()
                        info.update({
                            "coupon_closed": True,
                            "emergency_close": False,
                            "coupon_reward": float(self.total_reward - before_reward),
                            "coupon_max": float(self.total_max_points - before_max),
                            "coupon_hits": int(self.global_correct - before_hits),
                            "coupon_bets": int(self.global_bets - before_bets),
                            "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                            "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                            "perfect_coupon": bool(
                                getattr(self, "_last_coupon_total", 0) > 0 and
                                getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                            ),
                        })
                    else:
                        reward += self._emergency_close()
                        info.update(self._last_close_info)
                    reward, info, terminated = self._after_coupon_closed(reward, info)
                info = self._inject_common_info(info)
                return self._get_observation(), reward, terminated, False, info

        if close_flag:
            if self.min_bets_to_close <= self.bets_in_coupon <= self.max_bets_to_close and self.remaining_units >= self.coupon_price:
                before_reward = self.total_reward
                before_max = self.total_max_points
                before_hits = self.global_correct
                before_bets = self.global_bets
                reward += self._close_coupon()
                info.update({
                    "coupon_closed": True,
                    "emergency_close": False,
                    "coupon_reward": float(self.total_reward - before_reward),
                    "coupon_max": float(self.total_max_points - before_max),
                    "coupon_hits": int(self.global_correct - before_hits),
                    "coupon_bets": int(self.global_bets - before_bets),
                    "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                    "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                    "perfect_coupon": bool(
                        getattr(self, "_last_coupon_total", 0) > 0 and
                        getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                    ),
                })
                reward, info, terminated = self._after_coupon_closed(reward, info)
                info = self._inject_common_info(info)
                return self._get_observation(), reward, terminated, False, info

        weight = self.wide_weights.get(market_act, 1.0)
        if not is_unknown:
            correct = self._evaluate_prediction(
                int(h_ht), int(a_ht), int(h_goals), int(a_goals), market_act
            )
        else:
            correct = None

        if skip_flag == 1:
            if correct is True:
                delta = -self.skip_bad_penalty * weight
            elif correct is False:
                delta = +self.skip_good_bonus * weight
            else:
                delta = -0.2 * prior_score if prior_score >= 0.6 else 0.0
            reward += delta
            reason = ("decision_skip_good_incorrect" if delta > 0 else
                      ("decision_skip_bad_correct" if correct is True else
                       ("decision_skip_highprior" if delta < 0 and prior_score >= 0.6 else "decision_skip_unknown")))
            self._skip_match(idx, reason, delta)

            if not self._draw_next_from_coupon():
                if self.bets_in_coupon >= self.min_bets_to_close and self.remaining_units >= self.coupon_price:
                    before_reward = self.total_reward
                    before_max = self.total_max_points
                    before_hits = self.global_correct
                    before_bets = self.global_bets
                    reward += self._close_coupon()
                    info.update({
                        "coupon_closed": True,
                        "emergency_close": False,
                        "coupon_reward": float(self.total_reward - before_reward),
                        "coupon_max": float(self.total_max_points - before_max),
                        "coupon_hits": int(self.global_correct - before_hits),
                        "coupon_bets": int(self.global_bets - before_bets),
                        "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                        "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                        "perfect_coupon": bool(
                            getattr(self, "_last_coupon_total", 0) > 0 and
                            getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                        ),
                    })
                else:
                    reward += self._emergency_close()
                    info.update(self._last_close_info)
                reward, info, terminated = self._after_coupon_closed(reward, info)
            info = self._inject_common_info(info)
            return self._get_observation(), reward, terminated, False, info

        # --- Dodawanie do kuponu + STREAK LOGIKA (kupon + global) ---
        missed = False
        if is_unknown:
            self.coupon.append((idx, market_act, None, weight, False, risk_flag))
            for k in keys_now:
                self._coupon_taken_keys.add(k)
            topk = set(np.argsort(priors)[-self.prior_topk:])
            if market_act in topk:
                reward += self.prior_align_bonus_unknown * weight
        else:
            ok = bool(correct)
            missed = not ok
            self.coupon.append((idx, market_act, ok, weight, True, risk_flag))
            for k in keys_now:
                self._coupon_taken_keys.add(k)

            if ok:
                reward += self.step_hit_bonus * weight * self.mode_hit_boost[risk_flag]
                if self._coupon_series_active:
                    self._coupon_streak += 1
                    self._coupon_longest_streak = max(self._coupon_longest_streak, self._coupon_streak)
                    # progressive (quadratic) bonus for growing streak within coupon
                    reward += self.streak_step_bonus * (self._coupon_streak ** 2)
            else:
                reward -= (self.step_miss_pen / max(weight, 1e-6)) * self.mode_miss_boost[risk_flag]
                if self._coupon_series_active:
                    self._coupon_series_active = False

            if ok:
                self._global_streak += 1
                self._global_longest_streak = max(self._global_longest_streak, self._global_streak)
                # progressive (quadratic) bonus for global streak across coupons
                reward += self.global_streak_step_bonus * (self._global_streak ** 2)
            else:
                self._global_streak = 0

            topk = set(np.argsort(priors)[-self.prior_topk:])
            if ok and (market_act in topk):
                reward += self.prior_align_bonus_known * weight * (1.1 if is_sure else 1.0)

            # aux-consistency (delikatny)
            win_bias = (self.meta[idx].get("prior_ph", 1/3) - self.meta[idx].get("prior_pa", 1/3))
            likely_1 = win_bias >  0.10
            likely_2 = win_bias < -0.10
            eg = max(0.0, self.meta[idx].get("prior_gh", 1.1)) + max(0.0, self.meta[idx].get("prior_ga", 1.0))
            likely_o25 = eg > 2.6
            likely_btts = min(self.meta[idx].get("prior_gh",0.0), self.meta[idx].get("prior_ga",0.0)) > 0.9

            consistent = (
                (market_act == 0 and likely_1) or
                (market_act == 1 and likely_2) or
                (market_act == 7 and likely_o25) or
                (market_act == 5 and likely_btts)
            )
            if consistent:
                reward += self.aux_consistency_coef

        if id_fp:
            self.used_match_ids.add(id_fp)

        prev_div = self._coupon_diversity()
        self.market_counts[market_act] += 1
        new_div = self._coupon_diversity()
        reward += self.diversity_step_bonus * self.mode_div_bonus[risk_flag] * max(0.0, new_div - prev_div)

        ratio = float(self.market_counts[market_act]) / float(max(1, self.market_counts.sum()))
        if ratio > self.max_same_market_ratio:
            reward -= self.same_market_penalty * (ratio - self.max_same_market_ratio) * 10.0

        self.bets_in_coupon += 1

        if missed:
            if self.remaining_units >= self.coupon_price:
                before_reward = self.total_reward
                before_max = self.total_max_points
                before_hits = self.global_correct
                before_bets = self.global_bets
                reward += self._close_coupon()
                info.update({
                    "coupon_closed": True,
                    "emergency_close": False,
                    "coupon_reward": float(self.total_reward - before_reward),
                    "coupon_max": float(self.total_max_points - before_max),
                    "coupon_hits": int(self.global_correct - before_hits),
                    "coupon_bets": int(self.global_bets - before_bets),
                    "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                    "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                    "perfect_coupon": bool(
                        getattr(self, "_last_coupon_total", 0) > 0 and
                        getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                    ),
                })
                reward, info, terminated = self._after_coupon_closed(reward, info)
                info = self._inject_common_info(info)
                return self._get_observation(), reward, terminated, False, info
            else:
                info = self._inject_common_info(info)
                return self._get_observation(), reward, True, False, info

        if self.bets_in_coupon >= self.max_bets_to_close:
            if self.remaining_units >= self.coupon_price:
                before_reward = self.total_reward
                before_max = self.total_max_points
                before_hits = self.global_correct
                before_bets = self.global_bets
                reward += self._close_coupon()
                info.update({
                    "coupon_closed": True,
                    "emergency_close": False,
                    "coupon_reward": float(self.total_reward - before_reward),
                    "coupon_max": float(self.total_max_points - before_max),
                    "coupon_hits": int(self.global_correct - before_hits),
                    "coupon_bets": int(self.global_bets - before_bets),
                    "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                    "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                    "perfect_coupon": bool(
                        getattr(self, "_last_coupon_total", 0) > 0 and
                        getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                    ),
                })
                reward, info, terminated = self._after_coupon_closed(reward, info)
                info = self._inject_common_info(info)
                return self._get_observation(), reward, terminated, False, info
            else:
                info = self._inject_common_info(info)
                return self._get_observation(), reward, True, False, info

        if not self._draw_next_from_coupon():
            if self.bets_in_coupon >= self.min_bets_to_close and self.remaining_units >= self.coupon_price:
                before_reward = self.total_reward
                before_max = self.total_max_points
                before_hits = self.global_correct
                before_bets = self.global_bets
                reward += self._close_coupon()
                info.update({
                    "coupon_closed": True,
                    "emergency_close": False,
                    "coupon_reward": float(self.total_reward - before_reward),
                    "coupon_max": float(self.total_max_points - before_max),
                    "coupon_hits": int(self.global_correct - before_hits),
                    "coupon_bets": int(self.global_bets - before_bets),
                    "coupon_total": int(getattr(self, "_last_coupon_total", 0)),
                    "coupon_correct": int(getattr(self, "_last_coupon_correct", 0)),
                    "perfect_coupon": bool(
                        getattr(self, "_last_coupon_total", 0) > 0 and
                        getattr(self, "_last_coupon_total", 0) == getattr(self, "_last_coupon_correct", 0)
                    ),
                })
            else:
                reward += self._emergency_close()
                info.update(self._last_close_info)
            reward, info, terminated = self._after_coupon_closed(reward, info)

        info = self._inject_common_info(info)
        return self._get_observation(), reward, terminated, False, info

    def _coupon_diversity(self) -> float:
        total = self.market_counts.sum()
        if total <= 1:
            return 0.0
        p = self.market_counts / total
        return float(1.0 - np.sum(p * p))

    def _close_coupon(self) -> float:
        if not self.coupon:
            return 0.0

        total_all = len(self.coupon)
        total_known = sum(1 for *_, known, _risk in self.coupon if known)
        known_frac = (total_known / max(1, total_all))
        cost = self.coupon_price * max(1, known_frac)
        self.coupon_cost = cost

        base = match_bonus = wrong_pen = 0.0
        weighted_hits = weighted_bets = 0.0
        correct_known = 0

        for _, act, ok, w, known, risk_flag in self.coupon:
            if known:
                weighted_bets += w
                if ok:
                    base        += w * self.base_per_correct * self.mode_hit_boost[risk_flag]
                    match_bonus += w * self.match_bonus_per_correct
                    weighted_hits += w
                    correct_known += 1
                else:
                    wrong_pen   += (self.wrong_penalty / max(w, 1e-6)) * self.mode_miss_boost[risk_flag]

        self.global_bets    += int(total_known)
        self.global_correct += int(correct_known)

        hit_bonus    = weighted_hits ** 2
        length_bonus = weighted_bets * self.length_bonus_per_bet
        length_boost = (total_known >= 5) * (total_known ** 2)
        non_linear   = ((weighted_hits / total_known) ** 2) * total_known * 8 if total_known else 0.0

        raw = (base + hit_bonus + match_bonus +
               length_bonus + length_boost + non_linear - wrong_pen)

        raw += self.streak_close_bonus * (self._coupon_longest_streak ** 2)

        if total_known > 0:
            if correct_known == total_known:
                raw += self.perfect_bonus_per_cost * cost
            else:
                raw -= 2 * cost
        else:
            raw -= 1 * cost

        reward = max(0.0, raw)

        if total_known > 0:
            perfect_raw = (total_known * self.base_per_correct + total_known**2 +
                           self.match_bonus_per_correct * total_known + total_known * 8 +
                           length_bonus + length_boost +
                           self.perfect_bonus_per_cost * cost +
                           self.diversity_close_bonus * (1.0 - 1.0/self.num_markets) * total_known)
        else:
            perfect_raw = 0.0
        max_pts = perfect_raw

        self.remaining_units -= cost
        self.total_reward    += reward
        self.total_max_points += max_pts

        self._last_coupon_total = int(total_known)
        self._last_coupon_correct = int(correct_known)

        self._render(cost, reward, correct_known, total_known, max_pts)

        self.coupon.clear()
        self.bets_in_coupon = 0
        self._coupon_streak = 0
        self._coupon_longest_streak = 0
        self._coupon_series_active = True
        self._coupon_taken_keys.clear()

        return reward

    def _render(self, cost: float, reward: float, correct: int, total: int, max_pts: float) -> None:
        used = self.total_units - self.remaining_units
        print(f"\n🎯 Użyto: {used:.2f}/{self.total_units:.2f} jednostek")
        bet_names = self.BET_NAMES

        def _safe(txt: Any, fallback: str) -> str:
            if txt is None: return fallback
            if isinstance(txt, float) and np.isnan(txt): return fallback
            s = str(txt).strip()
            return s if s else fallback

        def _fmt_goal(x: Any) -> str:
            if x is None or (isinstance(x, float) and np.isnan(x)): return "?"
            try: return str(int(x)) if x == int(x) else str(x)
            except (ValueError, TypeError): return "?"

        def _fmt_datetime(raw_ts):
            try:
                if raw_ts is None: return None
                ts = pd.to_datetime(raw_ts, errors="coerce")
                if pd.notna(ts):
                    if getattr(ts, "tzinfo", None) is None:
                        ts = ts.tz_localize("UTC")
                    return ts.tz_convert("Europe/Warsaw").strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
            return None

        n_features = int(self.X.shape[1]) if isinstance(self.X, np.ndarray) else None

        for idx, act, ok, w, known, risk_flag in sorted(self.coupon, key=lambda t: self._sort_key(t[0])):
            if not (0 <= idx < len(self.y)): continue
            m = self.meta[idx]
            home = _safe(m.get("home_team"), "HOME")
            away = _safe(m.get("away_team"), "AWAY")
            h, a = self.y[idx]
            icon = "🔵" if not known else ("🟢" if ok else "🔴")
            risk = "P" if risk_flag == 0 else "V"
            bet_display = bet_names[act] if 0 <= act < len(bet_names) else f"A{act}"
            match_datetime = _fmt_datetime(m.get("start_time"))

            print(f"{icon} [{risk}] {home} vs {away} ({_fmt_goal(h)}-{_fmt_goal(a)}), "
                  f"zakład: {bet_display}, OK: {ok if known else 'unknown'}, waga: {w:.2f}, "
                  f"data i czas: {match_datetime}, cechy: {n_features}")

        w_total = sum(w for *_, w, known, _risk in self.coupon if known)
        w_hits  = sum(w for *_, ok, w, known, _risk in self.coupon if known and ok)
        pct     = 100.0 * w_hits / w_total if w_total else 0.0
        print(f"💰 Koszt kuponu: {cost:.2f}, Punkty: {reward:.0f}/{max_pts:.0f}  "
              f"(trafione wagi: {w_hits:.2f}/{w_total:.2f}  ⇒  {pct:.1f}%)")
        print(f"📏 Seria kuponu: {self._coupon_streak} | Global: {self._global_streak} "
              f"(max {self._global_longest_streak})")

        if self.skipped:
            print("\n⏭ Skipowane mecze:")
            for idx, reason, delta in self.skipped:
                if not (0 <= idx < len(self.meta)):
                    continue
                m = self.meta[idx]
                home = _safe(m.get("home_team"), "HOME")
                away = _safe(m.get("away_team"), "AWAY")
                match_datetime = _fmt_datetime(m.get("start_time"))
                efekt = "nagroda" if delta > 0 else ("kara" if delta < 0 else "neutral")
                print(f"   - {home} vs {away} ({match_datetime}), powód: {reason}, {efekt}: {delta:+.2f}")

    def render(self, mode="human"):
        bet_names = self.BET_NAMES
        total = self.market_counts.sum()
        hist = (self.market_counts / max(total, 1)).round(2)
        named_hist = {
            bet_names[i]: float(hist[i])
            for i in range(min(len(hist), len(bet_names)))
        }
        print(
            f"📦 Kupon: {self.bets_in_coupon} betów | Seria: {self._coupon_streak} | "
            f"Użyto {self.total_units - self.remaining_units:.1f}/{self.total_units:.1f} j. | "
            f"Global hits: {self.global_correct}/{self.global_bets} | "
            f"Global streak: {self._global_streak} (max {self._global_longest_streak})"
        )
        print(f"📈 Histogram rynków: {named_hist}")
        if self.skipped:
            print("⏭ Ostatnie skipy:")
            for idx, reason, delta in self.skipped[-5:]:
                efekt = "nagroda" if delta > 0 else ("kara" if delta < 0 else "neutral")
                print(f"   • idx={idx} → {reason} | {efekt}: {delta:+.2f}")
        else:
            print("⏭ Skipy: (brak)")

# ====== PREPROCESSOR z unii kolumn (FIX) ======
def build_global_preprocessor_from_batches(
    batches: list[pd.DataFrame],
    sample_per_batch: int = 1000
) -> PreprocType:
    frames = []
    for df in batches:
        if df.empty:
            continue
        df_samp = df.sample(min(len(df), sample_per_batch), random_state=42)
        df_samp = add_missing_columns(df_samp)
        frames.append(df_samp)
    if not frames:
        raise ValueError("Brak danych do zbudowania preprocesora.")

    df_all = pd.concat(frames, ignore_index=True)
    META_COLS = {"home_team", "away_team", "id_fp"}
    df_all = _numeric_coerce(df_all, META_COLS)

    DROP_COLS = [
        "fixture_id", "id_pred", "fetched_at",
        "payload_h2h", "payload_teams_away_logo", "payload_teams_home_logo",
        "payload_teams_away_league_lineups", "payload_teams_home_league_lineups",
        "payload_league_flag", "payload_league_logo",
        "score_fulltime_home", "score_fulltime_away", "score_halftime_home", "score_halftime_away"
        ]
    df_feat = df_all.drop(columns=[c for c in DROP_COLS if c in df_all.columns], errors="ignore").copy()
    _convert_start_time_inplace(df_feat)

    preproc = build_preprocessor(df_feat)
    preproc.fit(df_feat)
    return preproc

# ==============================
#  ROBUST LOADER (naprawa .load)
# ==============================
def robust_load_sb3(ckpt_path: str, env, device: str, fallback_policy_kwargs: dict | None = None):
    try:
        model = PPO.load(ckpt_path, env=env, device=device)
        print("🧠 Załadowano pełny checkpoint SB3 (PPO.load).")
        return model
    except Exception as e:
        print(f"ℹ️ PPO.load nieudane (OK, próbuję ręcznie): {e}")

    with zipfile.ZipFile(ckpt_path) as z:
        files = set(z.namelist())
        has_data_pkl  = "data.pkl" in files
        has_data_raw  = "data" in files
        has_policy    = "policy.pth" in files
        has_params    = "parameters.pt" in files
        has_pt_vars   = "pytorch_variables.pth" in files

        if "_stable_baselines3_version" in files:
            try:
                print("ℹ️ Checkpoint trenowany na SB3=", z.read("_stable_baselines3_version").decode().strip())
            except Exception:
                pass

        meta = None
        if has_data_pkl or has_data_raw:
            try:
                with z.open("data.pkl" if has_data_pkl else "data", "r") as f:
                    meta = cp.load(f)
            except Exception as e:
                print(f"⚠️ Nie udało się odczytać meta z 'data(.pkl)': {e}")

        if meta is not None:
            policy_kwargs = dict(meta.get("policy_kwargs", {}))
            policy_kwargs["ortho_init"] = False
            model = PPO(
                policy=meta.get("policy_class", "MlpPolicy"),
                env=env,
                policy_kwargs=policy_kwargs,
                n_steps=meta.get("n_steps", 2048),
                gamma=meta.get("gamma", 0.99),
                device=device,
                verbose=0,
                )
            try:
                state = None
                if has_policy:
                    state = torch.load(io.BytesIO(z.read("policy.pth")), map_location=device)
                elif has_params:
                    state = torch.load(io.BytesIO(z.read("parameters.pt")), map_location=device)
                if state is not None:
                    missing, unexpected = model.policy.load_state_dict(state, strict=False)
                    print(f"🧩 Wczytano wagi policy: missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception as e:
                print(f"⚠️ Nie udało się wgrać wag policy: {e}")

            if has_pt_vars:
                print("ℹ️ pytorch_variables.pth obecne (niepotrzebne do inferencji).")
            print("✅ Odtworzono model z meta ('data'/'data.pkl').")
            return model

        if not fallback_policy_kwargs:
            raise RuntimeError(
                "Brak pliku 'data'/'data.pkl' i brak fallback_policy_kwargs. "
                "Podaj POLICY_KWARGS z treningu."
                )

        policy_kwargs = dict(fallback_policy_kwargs)
        policy_kwargs["ortho_init"] = False
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            )
        if has_policy or has_params:
            try:
                state = torch.load(io.BytesIO(z.read("policy.pth" if has_policy else "parameters.pt")),
                                   map_location=device)
                missing, unexpected = model.policy.load_state_dict(state, strict=False)
                print(f"🧩 Wczytano wagi (weights-only): missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception as e:
                print(f"⚠️ Nie udało się wgrać wag z weights-only: {e}")
        else:
            print("ℹ️ Brak wag w ZIP (policy.pth/parameters.pt). Powstał świeży model.")
        return model


# ==============================
#  PIPELINE
# ==============================

torch.backends.cudnn.benchmark = True

def main() -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    BASE_DIR        = "/content/drive/MyDrive/STADIONY_DANE/NOWA_ERA"
    BEST_MODEL_PATH = f"{BASE_DIR}/best_model.zip"
    PREPROCESS_PATH = f"{BASE_DIR}/preprocess.pkl"
    VECNORM_PATH    = f"{BASE_DIR}/vecnormalize.pkl"
    LOG_DIR         = f"{BASE_DIR}/logs"
    # Bezpieczne tworzenie katalogów bazowych
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    DAILY_LIMIT = 2_000
    N_STEPS_PER_DAY = 4096
    EVAL_RUNS_PER_DAY = 10
    EVAL_DETERMINISTIC = False  # shadow-eval

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GLOBAL_SEED = 42
    set_random_seed(GLOBAL_SEED)

    POLICY_KWARGS = dict(
        net_arch=[8096, 4096, 2048, 1024, 512, 512, 256],
        activation_fn=nn.SiLU,
        ortho_init=False,
    )

    LEARNING_RATE = cosine_schedule(1e-4, 3e-5)
    ENT_CONST = 0.1
    TARGET_KL = 0.015
    CLIP_RANGE = 0.20

    def min_bets_for_pass(pass_idx: int) -> int:
        return min(38, 35 + 2 * (pass_idx - 1))

    def max_bets_for_pass(pass_idx: int) -> int:
        return 40

    # Curriculum: łagodniejsze progi i łagodniejsze zaostrzanie
    def prior_gates_for_pass(pass_idx: int) -> tuple[float, float]:
        base_sure, base_value = 0.44, 0.00
        inc = min(0.10, 0.015*(pass_idx-1))
        return base_sure + inc, base_value + min(inc, 0.08)

    if "one_day_batches" not in globals() or not one_day_batches:
        raise RuntimeError("⚠️  Zmienna one_day_batches jest pusta lub nie istnieje.")

    preprocess: PreprocType | None = None
    best_score = -1.0
    best_hit_pct = -1.0
    model: PPO | None = None

    # ——— PREPROCESSOR ———
    if os.path.exists(PREPROCESS_PATH):
        try:
            preprocess = joblib.load(PREPROCESS_PATH)
            print(f"🧠 Wczytano preprocesor z {PREPROCESS_PATH}")
        except Exception as e:
            print(f"⚠️  Błąd wczytywania preprocesora: {e}.")
            preprocess = None
    if preprocess is None:
        print("✨ Buduję globalny preprocesor (unia kolumn z wszystkich dni)…")
        preprocess = build_global_preprocessor_from_batches(one_day_batches, sample_per_batch=1200)
        os.makedirs(os.path.dirname(PREPROCESS_PATH), exist_ok=True)
        joblib.dump(preprocess, PREPROCESS_PATH)
        print(f"🖴  Zapisano preprocesor → {PREPROCESS_PATH}")

    # ——— OKNA WEEKENDOWE ———
    windows = build_weekend_windows(one_day_batches)
    if not windows:
        print("⚠️  Brak okna Fri+Sat / Sat+Sun – odpalam na wszystkich batchach.")
        selected_windows = [("all", one_day_batches, (None, None))]
    else:
        selected_windows = windows
        info_lines = []
        for name, batches_ws, (d0, d1) in selected_windows:
            label = "Piątek+Sobota" if name == "fri_sat" else "Sobota+Niedziela"
            d0s = f"{d0.date()} ({d0.day_name()})" if d0 is not None else "?"
            d1s = f"{d1.date()} ({d1.day_name()})" if d1 is not None else "?"
            info_lines.append(f"• {label}: {d0s} → {d1s} ({len(batches_ws)} batchy)")
        print("📅 Okna (Europe/Warsaw):\n  " + "\n  ".join(info_lines))

    # ——— PĘTLA UCZENIA/EWALUACJI ———
    no_improve = 0
    for pass_idx in range(1, MAX_PASSES + 1):
        print(f"\n==================== PRZEBIEG #{pass_idx} (wszystkie okna) ====================")
        best_score_before = best_score
        best_hit_before = best_hit_pct

        pass_all_days_all_runs_perfect = True

        sure_gate, value_gate = prior_gates_for_pass(pass_idx)

        for name, batches_ws, (_d0, _d1) in selected_windows:
            label = {"fri_sat": "Piątek+Sobota", "sat_sun": "Sobota+Niedziela"}.get(name, "Wszystkie dni")
            print(f"\n——— Okno: {label} (Pass #{pass_idx}) ———")

            for day_idx, raw_df in enumerate(batches_ws, start=1):
                if raw_df.empty:
                    print(f"📭 Dzień {day_idx}: pusty batch – pomijam")
                    continue

                if len(raw_df) > DAILY_LIMIT:
                    raw_df = raw_df.sample(DAILY_LIMIT, random_state=GLOBAL_SEED + day_idx)

                # Sortowanie chronologiczne (Europe/Warsaw) z tie-breakerem
                if "start_time" in raw_df.columns:
                    try:
                        st = pd.to_datetime(raw_df["start_time"], errors="coerce")
                        if st.notna().any():
                            st_local = st.dt.tz_localize(
                                "UTC", nonexistent="shift_forward", ambiguous="NaT", errors="ignore"
                            )
                            st_local = st_local.dt.tz_convert("Europe/Warsaw")
                            key_time = st_local.view("int64")
                            home = raw_df.get("home_team", pd.Series([""]*len(raw_df))).astype(str).str.lower()
                            away = raw_df.get("away_team", pd.Series([""]*len(raw_df))).astype(str).str.lower()
                            idfp = raw_df.get("id_fp", pd.Series([""]*len(raw_df))).astype(str)
                            ord_df = pd.DataFrame({
                                "_t": key_time,
                                "_h": home,
                                "_a": away,
                                "_i": idfp,
                            })
                            raw_df = raw_df.assign(_t=ord_df["_t"], _h=ord_df["_h"], _a=ord_df["_a"], _i=ord_df["_i"])
                            raw_df = raw_df.sort_values(by=["_t","_h","_a","_i"], kind="mergesort")
                            raw_df = raw_df.drop(columns=["_t","_h","_a","_i"])
                    except Exception:
                        pass

                # Przygotowanie batcha
                X_day, y_day, meta_day, preprocess, skip_mask = prepare_batch(
                    raw_df, preprocess, fit_if_none=False, save_path=None
                )
                if preprocess is None:
                    print(f"⚠️  Dzień {day_idx}: Preprocesor nie gotowy – pomijam.")
                    continue

                mn = min_bets_for_pass(pass_idx)
                mx = max_bets_for_pass(pass_idx)

                def make_thunk(X=X_day, Y=y_day, M=meta_day, SK=skip_mask,
                               seed=GLOBAL_SEED + day_idx, mn=mn, mx=mx,
                               sure_gate=sure_gate, value_gate=value_gate):
                    def _thunk():
                        base_env = StadiumMatchEnv(
                            X, Y, M,
                            total_units=20.0,
                            coupon_price=2.0,
                            min_bets_to_close=mn,
                            max_bets_to_close=mx,
                            skip_mask=SK,
                            seed=seed,
                            min_prior_for_bet=0.65,
                            auto_skip_penalty_coef=0.2,
                        )
                        # curriculum gates:
                        base_env.min_prior_sure  = float(sure_gate)
                        base_env.min_prior_value = float(value_gate)
                        return ImputeNaNWrapper(base_env, impute_value=0.0)
                    return _thunk

                vec_env = DummyVecEnv([make_thunk()])

                # Jeden VecNormalize wspólny (utrzymuj spójny plik)
                if os.path.exists(VECNORM_PATH):
                    try:
                        vecnorm = VecNormalize.load(VECNORM_PATH, vec_env)
                        loaded_shape = int(np.prod(vecnorm.obs_rms.mean.shape))
                        env_shape = int(np.prod(vec_env.observation_space.shape))
                        if loaded_shape != env_shape:
                            print(
                                f"⚠️  Niezgodny kształt VecNormalize ({loaded_shape} ≠ {env_shape}) — inicjalizuję nowe statystyki."
                            )
                            vecnorm = VecNormalize(
                                vec_env,
                                norm_obs=True,
                                norm_reward=True,
                                clip_obs=10.0,
                                clip_reward=10.0,
                                gamma=0.995,
                            )
                        else:
                            print("📥 Załadowano VecNormalize.")
                    except Exception as e:
                        print(f"⚠️  Błąd load VecNormalize: {e} — inicjalizuję nowe statystyki.")
                        vecnorm = VecNormalize(
                            vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.995
                        )
                else:
                    vecnorm = VecNormalize(
                        vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.995
                    )

                # Model
                if model is None:
                    if os.path.exists(BEST_MODEL_PATH):
                        model = robust_load_sb3(BEST_MODEL_PATH, vecnorm, DEVICE, POLICY_KWARGS)
                    else:
                        print("✨ Inicjalizacja nowego modelu…")
                        model = PPO(
                            "MlpPolicy",
                            vecnorm,
                            learning_rate=LEARNING_RATE,
                            batch_size=4096,
                            n_steps=16384,
                            gae_lambda=0.98,
                            gamma=0.995,
                            clip_range=CLIP_RANGE,
                            policy_kwargs=POLICY_KWARGS,
                            verbose=1,
                            device=DEVICE,
                            ent_coef=ENT_CONST,
                            target_kl=TARGET_KL,
                            max_grad_norm=0.5,
                            vf_coef=1.0,
                        )
                else:
                    model.set_env(vecnorm)

                # Trening
                print(f"\n🚀 Dzień {day_idx}: trening na {len(X_day)} przykładach…")
                try:
                    vecnorm.training = True
                    vecnorm.norm_reward = True
                    model.learn(
                        total_timesteps=N_STEPS_PER_DAY,
                        reset_num_timesteps=False,
                        progress_bar=False,
                        callback=PolicyMarginCallback()
                    )
                    vecnorm.training = False
                    vecnorm.norm_reward = False  # ewaluacja bez normowania nagród
                    vecnorm.save(VECNORM_PATH)
                except Exception as e:
                    print(f"❌ Błąd treningu w dniu {day_idx}: {e}")
                    continue

                # Ewaluacja (shadow-eval)
                pts_sum = 0.0
                max_sum = 0.0
                hits_sum = 0
                bets_sum = 0

                print(f"🔬 Dzień {day_idx}: ewaluacja ({EVAL_RUNS_PER_DAY} budżetów, "
                      f"{'deterministic' if EVAL_DETERMINISTIC else 'stochastic'})…")
                perfect_runs_flags: List[bool] = []
                budget_stats: List[Tuple[int, bool]] = []

                for r in range(EVAL_RUNS_PER_DAY):
                    seen_perfect_this_run = False
                    coupons_closed_this_run = 0
                    try:
                        vecnorm.training = False
                        vecnorm.norm_reward = False
                        obs = vecnorm.reset()
                        done = np.array([False])
                        while not bool(done[0]):
                            action, _ = model.predict(obs, deterministic=EVAL_DETERMINISTIC)
                            # Upewnij się, że akcja ma 4 komponenty (w razie starych checkpointów)
                            if isinstance(action, np.ndarray) and action.shape[-1] == 3:
                                # dodaj risk_flag=0 (pewniak) domyślnie
                                action = np.concatenate([action, np.array([[0]], dtype=action.dtype)], axis=-1)
                            obs, rwd, done, infos = vecnorm.step(action)
                            info = infos[0] if isinstance(infos, list) else infos
                            if info and info.get("coupon_closed"):
                                coupons_closed_this_run += 1
                                pts_sum += float(info.get("coupon_reward", 0.0))
                                max_sum += float(info.get("coupon_max", 0.0))
                                hits_sum += int(info.get("coupon_hits", 0))
                                bets_sum += int(info.get("coupon_bets", 0))
                                if bool(info.get("perfect_coupon", False)):
                                    seen_perfect_this_run = True
                    except Exception as e:
                        print(f"❌ Błąd ewaluacji (budżet {r+1}): {e}")
                    perfect_runs_flags.append(seen_perfect_this_run)
                    budget_stats.append((coupons_closed_this_run, seen_perfect_this_run))

                day_score = (pts_sum / max_sum * 100) if max_sum > 0 else 0.0
                hit_pct   = (100.0 * hits_sum / bets_sum) if bets_sum > 0 else 0.0
                runs_perfect = int(sum(perfect_runs_flags))
                all_runs_perfect = (runs_perfect == EVAL_RUNS_PER_DAY)

                print("📊 Podsumowanie budżetów (runów) tego dnia:")
                for i, (cnt, okp) in enumerate(budget_stats, start=1):
                    print(f"   • Budżet {i}: kuponów={cnt:2d}, perfect={'TAK' if okp else 'NIE'}")

                print(f"✅ Skuteczność: {day_score:.1f}% | Trafienia: {hits_sum}/{bets_sum} ⇒ {hit_pct:.1f}% "
                      f"| Budżety z perfect: {runs_perfect}/{EVAL_RUNS_PER_DAY}")

                # Log CSV
                day_date = _infer_local_date(raw_df)
                date_tag = day_date.strftime("%Y%m%d") if day_date is not None else f"idx{day_idx:03d}"
                df_log = pd.DataFrame({
                    "pass_idx": [pass_idx] * len(budget_stats),
                    "window": [name] * len(budget_stats),
                    "day_idx": [day_idx] * len(budget_stats),
                    "run": list(range(1, len(budget_stats) + 1)),
                    "coupons_closed": [c for (c, _) in budget_stats],
                    "perfect": [bool(p) for (_, p) in budget_stats],
                })
                df_log["day_score_pct"] = day_score
                df_log["hit_pct"] = hit_pct

                csv_name = f"eval_budgets_pass{pass_idx}_{name}_{date_tag}.csv"
                csv_path = os.path.join(LOG_DIR, csv_name)
                df_log.to_csv(csv_path, index=False)
                print(f"📝 Zapisano log budżetów do: {csv_path}")

                # Update best & zapis modelu
                if day_score > best_score or hit_pct > best_hit_pct:
                    best_score = max(best_score, day_score)
                    best_hit_pct = max(best_hit_pct, hit_pct)
                    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
                    try:
                        model.save(BEST_MODEL_PATH)
                        print(f"💾 Zapisano nowy BEST_MODEL → {BEST_MODEL_PATH}")
                    except Exception as e:
                        print(f"⚠️  Nie udało się zapisać modelu: {e}")

                # czy w tym dniu wszystkie budżety miały perfect kupony?
                if not all_runs_perfect:
                    pass_all_days_all_runs_perfect = False

        # ——— po przejściu wszystkich okien w tym przebiegu ———
        improved = (best_score > best_score_before) or (best_hit_pct > best_hit_before)
        if improved:
            no_improve = 0
        else:
            no_improve += 1

        print(f"\n📈 Najlepsze do tej pory: day_score={best_score:.2f}% | hit_pct={best_hit_pct:.2f}%")
        if best_score >= TARGET_DAY_SCORE and best_hit_pct >= TARGET_HIT_PCT:
            print("🎉 Osiągnięto cele jakości — przerywam trening.")
            break

        if pass_all_days_all_runs_perfect:
            print("🏆 Wszystkie budżety w tym przebiegu miały perfect kupony — przerywam trening.")
            break

        if no_improve >= PATIENCE:
            print(f"⏹️  Brak poprawy przez {PATIENCE} przebiegi — przerywam trening.")
            break

    # ——— finalny wydruk ———
    print("\n==================== PODSUMOWANIE ====================")
    print(f"🥇 Najlepszy day_score: {best_score:.2f}%")
    print(f"🎯 Najlepszy hit_pct:  {best_hit_pct:.2f}%")
    print(f"📦 Model zapisany (jeśli poprawiał) w: {BEST_MODEL_PATH}")
    print(f"🧮 VecNormalize w: {VECNORM_PATH}")
    print("======================================================")

# if __name__ == "__main__":
#     main()
