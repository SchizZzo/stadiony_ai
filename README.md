# stadiony_ai

This repository now includes a simple utility for ranking match predictions by
probability and building the longest series of high-confidence picks, ignoring
dates and kick-off times. You can also generate an arbitrarily long series of
predictions simply by taking the top-N most confident matches.

## Match selector

`match_selector.py` demonstrates how to:

1. Store match predictions with probabilities.
2. Order matches from most to least probable.
3. Create a sequence of picks above a probability threshold.
4. Grab the top-N most confident matches for the longest possible streak.

Run the example:

```bash
python match_selector.py
```

The script prints each match with its kick-off time and probability,
ordered by confidence.

### RL integration

If you train a reinforcement-learning model (for example with
`stadiiony_szkolenie.py`), you can plug its policy into the selector. The
helper functions convert the model's action probabilities into
`MatchPrediction` objects and build the highest-confidence series:

```python
from datetime import datetime
from stable_baselines3 import PPO

from match_selector import rl_confident_series, rl_max_series

model = PPO.load("path/to/model.zip")
observations = [...]  # list of environment observations for each match
meta = [("Team A vs Team B", datetime(2025, 6, 1, 18, 0)), ...]

# Top matches above a fixed probability threshold
series = rl_confident_series(model, observations, meta, min_probability=0.6)

# Alternatively, simply take the five most confident predictions
top5 = rl_max_series(model, observations, meta, limit=5)
```

Both helpers return matches sorted purely by the model's confidence, ignoring
dates or kick-off times.
