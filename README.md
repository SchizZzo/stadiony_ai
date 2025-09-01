# stadiony_ai

This repository now includes a simple utility for ranking match predictions by
probability and building the longest series of high-confidence picks, ignoring
dates and kick-off times.

## Match selector

`match_selector.py` demonstrates how to:

1. Store match predictions with probabilities.
2. Order matches from most to least probable.
3. Create a sequence of picks above a probability threshold.

Run the example:

```bash
python match_selector.py
```

The script prints the matches in order of confidence.

### RL integration

If you train a reinforcement-learning model (for example with
`stadiiony_szkolenie.py`), you can plug its policy into the selector. The
helper functions convert the model's action probabilities into
`MatchPrediction` objects and build the highest-confidence series:

```python
from datetime import datetime
from stable_baselines3 import PPO

from match_selector import rl_confident_series

model = PPO.load("path/to/model.zip")
observations = [...]  # list of environment observations for each match
meta = [("Team A vs Team B", datetime(2025, 6, 1, 18, 0)), ...]

series = rl_confident_series(model, observations, meta, min_probability=0.6)
```

`series` now contains matches sorted purely by the model's confidence,
ignoring dates or kick-off times.
