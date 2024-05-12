import json

import pandas as pd
from pathlib import Path

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

base_models_path = Path("./model_checkpoints/")

model_paths = [pth for pth in base_models_path.iterdir()]

# keep_cols = ["model", "F1 Score", "Track 1: Classification Error",
#              "Track 2: Cost for Poisonousness Confusion",
#              "Track 3: User-Focused Loss"]

keep_cols = ["model", "Track 3: User-Focused Loss"]

performance_dfs = []
for model_path in model_paths:
    # score_result_path = model_path / "competition_metrics_scores.json"
    score_result_path = model_path / "competition_metrics_scores_opengan.json"
    if (score_result_path).exists():
        with open(score_result_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data, index=[0])
        df["model"] = score_result_path.parent.name
        performance_dfs.append(df[keep_cols])

performance_comparison_df = pd.concat(performance_dfs, ignore_index=True)
performance_comparison_df.drop_duplicates(inplace=True)
performance_comparison_df.reset_index(inplace=True, drop=True)

print(performance_comparison_df.sort_values("Track 3: User-Focused Loss", ascending=True))
