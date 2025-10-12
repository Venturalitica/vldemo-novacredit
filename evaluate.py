"""Stage `evaluate` (DVC): corre el eval agnóstico sobre el parquet del stage `featurize` y
persiste el modelo como out cacheado (Art.15). La medición (SDK) vive en compliance_eval; el
tratamiento (variante) en train.py."""

import joblib
import pandas as pd

import compliance_eval
import train

df = pd.read_parquet("data/features.parquet")
_, model = compliance_eval.run(train.build_model, df)
joblib.dump(model, "model.pkl")
