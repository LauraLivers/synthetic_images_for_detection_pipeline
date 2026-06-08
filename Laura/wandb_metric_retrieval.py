import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("llivers-hochschule-luzern/ladder-detection-optuna")

results = []
for run in runs:
    if run.state == "finished":
        history = run.history(keys=['metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)'])
        if not history.empty:
            best = history.loc[history['metrics/mAP50(B)'].idxmax()]
            results.append({
                'variant': run.config.get('dataset', run.name),
                'mAP50': best['metrics/mAP50(B)'],
                'precision': best['metrics/precision(B)'],
                'recall': best['metrics/recall(B)'],
            })

df = pd.DataFrame(results)
df = df.sort_values('mAP50', ascending=False)
print(df.to_latex(index=False))