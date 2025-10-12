"""Stage `featurize` de loan-scoring — carga el dataset REAL Statlog (German Credit) vía
Croissant (§2, no read_csv a mano) y produce `data/features.parquet` con `target`
(1=buen crédito/0=malo), `gender` (protegido, 1=male/0=female) y el resto de atributos
como features. Stage estable frente al tratamiento (el tratamiento cambia train.py)."""

import mlcroissant as mlc
import pandas as pd

CROISSANT = "data/german_credit.croissant.json"
RECORD_SET = "applications"  # @id del RecordSet en el Croissant de German Credit

ds = mlc.Dataset(jsonld=CROISSANT)
df = pd.DataFrame(list(ds.records(record_set=RECORD_SET)))
# mlcroissant nombra los campos "<recordset>/<col>" y devuelve texto en bytes.
df = df.rename(columns=lambda c: c.split("/", 1)[1] if "/" in c else c)
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)

# Etiqueta binaria ya viene derivada en el CSV: target (1=good/0=bad).
df["target"] = df["target"].astype(int)
# Atributo protegido para la paridad demográfica: gender (male/female) → 1/0.
df["gender"] = (df["gender"].astype(str).str.strip().str.lower() == "male").astype(int)
df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
# Quita la etiqueta cruda UCI `class` (fuga de la etiqueta) si está presente.
df = df.drop(columns=[c for c in ["class"] if c in df.columns])

df.to_parquet("data/features.parquet")
print("featurize → data/features.parquet")
