import numpy as np

# dict key should match name of function

def mae(y_true, y_pred, **kwargs):
    return {'mae': np.mean(np.abs(y_true - y_pred))}

def mae_patch(y_true, y_pred):
    percentiles = np.arange(5, 95, 5)
    maes = []
    for p in percentiles:
        agg = np.percentile(y_pred, p, axis=1)
        maes.append(np.mean(np.abs(y_true - agg)))
    return {
        'mae_patch': np.min(maes),
        'pct': percentiles[maes.index(np.min(maes))]
    }