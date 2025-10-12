"""Tratamiento de loan-scoring — modelo y su VARIANTE (V1/V2). Descarta el atributo
protegido `gender` de las features (no se decide con él). El arco honesto rojo→verde
viene de `mitigate`:

  mitigate=False (V1): LogisticRegression plano → puede heredar disparidad por gender.
  mitigate=True  (V2): fairlearn ExponentiatedGradient con restricción DemographicParity
                       → acota la paridad demográfica de la decisión (ISO 23894 §6.5,
                       venth.yaml treat: "Acotar la paridad demográfica por grupo protegido").

build_model(df, seed, mitigate) -> (cohort_con_prediction, modelo, X). El modelo mitigado
expone `_pmf_predict` (compliance_eval lo detecta para la decisión binaria)."""

import pandas as pd
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "target"
GENDER = "gender"


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de features: one-hot de todo menos el target, la predicción y el protegido. Incluye
    `class` (la etiqueta cruda UCI = codificación perfecta de target → fuga) por si la cohorte
    viene de la carga Croissant directa (featurize la elimina, pero el fallback no)."""
    leaky = ["class", TARGET, "prediction", GENDER]
    return pd.get_dummies(df.drop(columns=[c for c in leaky if c in df.columns]))


def build_model(df: pd.DataFrame, seed: int, mitigate: bool = False):
    # Escalado UP-FRONT (común a V1/V2): one-hot + StandardScaler ajustado sobre toda la cohorte,
    # determinista (sin estado por-grupo). Necesario para que la LR base converja sano y la
    # mitigación fairlearn (que reduce a clasificación con coste) opere sobre features estandarizadas.
    Xraw = _features(df).astype(float)
    scaler = StandardScaler(with_mean=False).fit(Xraw)
    X = pd.DataFrame(scaler.transform(Xraw), index=Xraw.index, columns=Xraw.columns)
    y = df[TARGET].astype(int)
    a = df[GENDER].astype(int)  # atributo sensible (solo para la mitigación, no es feature)
    Xtr, Xte, ytr, _, atr, _ = train_test_split(X, y, a, test_size=0.2, random_state=seed)
    base = LogisticRegression(max_iter=5000, random_state=seed)
    if mitigate:
        # In-processing fairlearn: la reducción pasa `sample_weight` al `fit` del estimador base.
        # Un sklearn Pipeline NO enruta `sample_weight` al paso final (rompe ExponentiatedGradient),
        # así que el base es un LogisticRegression PELADO (acepta sample_weight nativamente) — la
        # vía probada (tests/resources/loan/train_mitigated.py). eps=0.005 = restricción DP estricta.
        model = ExponentiatedGradient(base, constraints=DemographicParity(), eps=0.005)
        model.fit(Xtr, ytr, sensitive_features=atr)
    else:
        model = base.fit(Xtr, ytr)
    # M-11: la equidad del MODELO (Art.15, fase `validation`) se mide sobre el conjunto HELD-OUT
    # (test, 20%), NUNCA in-sample sobre las filas de entrenamiento. La fase `training` (Art.10,
    # equidad del DATO) sí usa la cohorte completa en compliance_eval — es la disparidad del dataset
    # crudo, no de la decisión del modelo.
    cohort = df.loc[Xte.index].copy()
    if hasattr(model, "_pmf_predict"):
        cohort["prediction"] = (model._pmf_predict(Xte)[:, 1] >= 0.5).astype(int)
    else:
        cohort["prediction"] = model.predict(Xte).astype(int)
    return cohort, model, X
