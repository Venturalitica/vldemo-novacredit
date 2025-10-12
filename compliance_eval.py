"""Núcleo del eval de NovaCredit (scoring crediticio within-appetite LIMPIO) — AGNÓSTICO al
framework MLOps (no importa DVC/MLflow). Carga el dato vía Croissant (§2), corre la SDK
venturalitica (vl.monitor abre la sesión + los probes —incl. BOMProbe—; vl.enforce evalúa el
OSCAL en dos fases Art.10/Art.15), PROMUEVE el bom.json del run a .venturalitica/bom.json (ruta
que lee el motor) y vuelca metrics.json plano {control_id: {value, power}}. NO juzga: el veredicto
autoritativo lo pone el motor Rust contra el MISMO OSCAL."""

import os

os.environ.setdefault("VENTURALITICA_NO_ANALYTICS", "1")  # sin telemetría en CI

import contextlib
import json
import shutil
import sys
from pathlib import Path

import mlcroissant as mlc
import pandas as pd
import yaml

import venturalitica as vl

CROISSANT = "data/german_credit.croissant.json"
RECORD_SET = "applications"
OSCAL = "shared_data/policies/assessment_plan.oscal.yaml"
PARAMS = "params.yaml"
METRICS = "metrics.json"
BOM_ROOT = ".venturalitica/bom.json"
RUNS_DIR = Path(".venturalitica/runs")

TARGET = "target"
GENDER = "gender"


def load_applications(croissant_path: str = CROISSANT) -> pd.DataFrame:
    ds = mlc.Dataset(jsonld=croissant_path)
    df = pd.DataFrame(list(ds.records(record_set=RECORD_SET)))
    df = df.rename(columns=lambda c: c.split("/", 1)[1] if "/" in c else c)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    df[TARGET] = df[TARGET].astype(int)
    df[GENDER] = (df[GENDER].astype(str).str.strip().str.lower() == "male").astype(int)
    # Quita la etiqueta cruda UCI `class` (codificación perfecta de target → fuga) en el path
    # Croissant directo; featurize ya la elimina en el path DVC (defensa en profundidad).
    df = df.drop(columns=[c for c in ["class"] if c in df.columns])
    return df


def params() -> dict:
    return yaml.safe_load(open(PARAMS)) or {}


def _control_order(oscal_path: str) -> dict:
    doc = yaml.safe_load(open(oscal_path))
    reqs = doc["component-definition"]["components"][0]["control-implementations"][0][
        "implemented-requirements"
    ]
    return {r["control-id"]: i for i, r in enumerate(reqs)}


def _metric_entry(result) -> float | dict:
    """Una entrada de `metrics.json`: objeto `{value, power}` si el SDK expone el bloque de poder
    (bootstrap, ≥0.6.11), escalar `value` si no. El núcleo Rust acepta ambas formas (untagged)."""
    value = float(result.actual_value)
    power = getattr(result, "power", None)
    return {"value": value, "power": power} if power else value


def _enforce_phases(df: pd.DataFrame, cohort: pd.DataFrame, oscal_path: str) -> dict:
    data_results = vl.enforce(
        data=df, policy=oscal_path, target=TARGET, gender=GENDER,
        phase="training", strict=False,
    )
    model_results = vl.enforce(
        data=cohort, policy=oscal_path, target=TARGET, prediction="prediction",
        gender=GENDER, phase="validation", strict=False,
    )
    order = _control_order(oscal_path)
    results = sorted(data_results + model_results, key=lambda r: order.get(r.control_id, 10**6))
    return {r.control_id: _metric_entry(r) for r in results}


def _promote_bom() -> None:
    """Promueve el bom.json que BOMProbe dejó en .venturalitica/runs/<run>/ a la raíz
    .venturalitica/bom.json (lo que read_bom lee, bom.rs:15). Elige el run con mtime máximo
    (misma heurística que el CLI push). FAIL-LOUD si no hay ningún bom.json que promover."""
    candidates = sorted(
        (p for p in RUNS_DIR.glob("*/bom.json") if p.parent.name != "latest"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("compliance_eval: no se generó ningún bom.json en .venturalitica/runs/")
    Path(".venturalitica").mkdir(exist_ok=True)
    shutil.copyfile(candidates[-1], BOM_ROOT)
    print(f"bom → {BOM_ROOT} (desde {candidates[-1]})", file=sys.stderr)


def write_metrics(metrics: dict, path: str = METRICS) -> None:
    json.dump(metrics, open(path, "w"), indent=2)


def run(build_model, df: pd.DataFrame | None = None, oscal_path: str = OSCAL):
    if df is None:
        df = load_applications()
    p = params()
    seed = int(p.get("seed", 42))
    mitigate = bool(p.get("mitigate", False))
    with contextlib.redirect_stdout(sys.stderr):
        with vl.monitor(name="novacredit", label="venth eval"):
            cohort, model, _ = build_model(df, seed, mitigate)  # ENTRENAMIENTO (Art.15)
            metrics = _enforce_phases(df, cohort, oscal_path)    # medición (Art.10/Art.15)
        _promote_bom()  # tras cerrar la sesión, el bom.json del run ya existe
    write_metrics(metrics)
    return cohort, model
