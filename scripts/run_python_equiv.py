#!/usr/bin/env python
"""Run csdid-python on the Medicaid replication sample and normalize outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csdid.att_gt import ATTgt


SEED = 20240924
BOOT_ITERS = 25_000
COVS: List[str] = [
    "perc_female",
    "perc_white",
    "perc_hispanic",
    "unemp_rate",
    "poverty_rate",
    "median_income",
]
DATA_PATH = Path(
    os.environ.get(
        "JEL_DID_DATA",
        "/Users/gabrielsaco/Documents/GitHub/JEL-DiD/data/county_mortality_data.csv",
    )
)
OUT_DIR = Path("report/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data() -> pd.DataFrame:
    """Mirror the R data prep used in Code_Appendix Table 7."""
    df = pd.read_csv(DATA_PATH)
    df["state"] = df["county"].str[-2:]
    df = df.loc[~df["state"].isin(["DC", "DE", "MA", "NY", "VT"])].copy()
    df = df.loc[
        (df["yaca"] == 2014) | df["yaca"].isna() | (df["yaca"] > 2019)
    ].copy()

    df = df.assign(
        perc_white=df["population_20_64_white"] / df["population_20_64"] * 100,
        perc_hispanic=df["population_20_64_hispanic"] / df["population_20_64"] * 100,
        perc_female=df["population_20_64_female"] / df["population_20_64"] * 100,
        unemp_rate=df["unemp_rate"] * 100,
        median_income=df["median_income"] / 1000,
    )

    keep_cols = list(
        dict.fromkeys(
            [
                "state",
                "county",
                "county_code",
                "year",
                "population_20_64",
                "yaca",
                *[c for c in df.columns if c.startswith("perc_")],
                "crude_rate_20_64",
                *COVS,
            ]
        )
    )
    df = df[keep_cols].copy()

    cols_to_check = [c for c in df.columns if c != "yaca"]
    df = df.dropna(subset=cols_to_check)

    df = (
        df.groupby("county_code")
        .filter(lambda g: ((g["year"] == 2013) | (g["year"] == 2014)).sum() == 2)
        .copy()
    )
    df = (
        df.groupby("county_code")
        .filter(lambda g: g["crude_rate_20_64"].notna().all() and len(g) == 11)
        .copy()
    )

    df["county_code"] = pd.to_numeric(df["county_code"])
    return df


def add_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 2013 population weight per county."""
    short = df.loc[df["year"].isin([2013, 2014])].copy()
    base_wt = (
        short.loc[short["year"] == 2013, ["county_code", "population_20_64"]]
        .rename(columns={"population_20_64": "set_wt"})
        .drop_duplicates(subset=["county_code"])
    )
    short = short.merge(base_wt, on="county_code", how="left")
    short["Treat"] = np.where((short["yaca"] == 2014) & short["yaca"].notna(), 1, 0)
    short["Post"] = (short["year"] == 2014).astype(int)
    short["treat_year"] = np.where(
        (short["yaca"] == 2014) & short["yaca"].notna(), 2014, 0
    )
    return short


def run_attgt(
    data: pd.DataFrame, method: str, weighted: bool
) -> pd.DataFrame:
    """Estimate ATT(g,t) with specified method and weight choice."""
    weights_name: Optional[str] = "set_wt" if weighted else None

    att = ATTgt(
        yname="crude_rate_20_64",
        tname="year",
        idname="county_code",
        gname="treat_year",
        data=data.copy(),
        control_group="nevertreated",
        xformla="~ " + " + ".join(COVS),
        panel=True,
        allow_unbalanced_panel=False,
        cband=True,
        weights_name=weights_name,
        biters=BOOT_ITERS,
    ).fit(est_method=method, base_period="universal", bstrap=True)

    att.aggte(typec="group", na_rm=True, biters=BOOT_ITERS)
    agg = att.atte
    crit = float(np.array(agg["crit_val_egt"]).ravel()[0])

    g_vals = np.array(agg["egt"]).astype(int).ravel()
    att_vals = np.array(agg["att_egt"]).astype(float).ravel()
    se_vals = np.array(agg["se_egt"]).astype(float).ravel()

    rows = []
    for g_val, est_val, se_val in zip(g_vals, att_vals, se_vals):
        ci_lo = est_val - crit * se_val
        ci_hi = est_val + crit * se_val
        rows.append(
            {
                "spec_id": "medicaid_table7",
                "method": method,
                "weighted": weighted,
                "agg_type": "group",
                "g": int(g_val),
                "t": int(g_val),
                "e": 0,
                "estimate": float(est_val),
                "se": float(se_val),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
            }
        )
    return pd.DataFrame(rows)


def compare_truth(truth_path: Path, py_path: Path) -> pd.DataFrame:
    """Compare Python output against truth and emit cell-by-cell diffs."""
    truth = pd.read_csv(truth_path)
    py = pd.read_csv(py_path)
    keys = ["spec_id", "method", "weighted", "agg_type", "g", "t", "e"]
    merged = truth.merge(py, on=keys, suffixes=("_truth", "_py"))

    atols = {"estimate": 1e-6, "se": 1e-6, "ci_lo": 1e-6, "ci_hi": 1e-6}
    rtols = {"estimate": 1e-6, "se": 1e-4, "ci_lo": 1e-6, "ci_hi": 1e-6}

    records = []
    for _, row in merged.iterrows():
        for metric in ["estimate", "se", "ci_lo", "ci_hi"]:
            truth_val = row[f"{metric}_truth"]
            py_val = row[f"{metric}_py"]
            abs_diff = float(abs(py_val - truth_val))
            denom = max(abs(truth_val), 1e-12)
            rel_diff = abs_diff / denom
            passed = np.isclose(
                py_val, truth_val, atol=atols[metric], rtol=rtols[metric]
            )
            records.append(
                {
                    **{k: row[k] for k in keys},
                    "metric": metric,
                    "truth": float(truth_val),
                    "python": float(py_val),
                    "abs_diff": abs_diff,
                    "rel_diff": float(rel_diff),
                    "pass": bool(passed),
                    "atol": atols[metric],
                    "rtol": rtols[metric],
                }
            )
    diff = pd.DataFrame.from_records(records)
    diff.to_csv(OUT_DIR / "diff.csv", index=False)
    return diff


def main() -> None:
    np.random.seed(SEED)
    data = prepare_data()
    short = add_weights(data)

    results = pd.concat(
        [
            run_attgt(short, "reg", weighted=False),
            run_attgt(short, "ipw", weighted=False),
            run_attgt(short, "dr", weighted=False),
            run_attgt(short, "reg", weighted=True),
            run_attgt(short, "ipw", weighted=True),
            run_attgt(short, "dr", weighted=True),
        ],
        ignore_index=True,
    )
    results.to_csv(OUT_DIR / "python.csv", index=False)

    meta = pd.DataFrame(
        [
            {
                "source": "python",
                "n_rows": len(short),
                "n_units": short["county_code"].nunique(),
                "treated_units": short.loc[short["treat_year"] == 2014, "county_code"]
                .nunique(),
                "control_units": short.loc[short["treat_year"] == 0, "county_code"]
                .nunique(),
                "weight_min": short["set_wt"].min(),
                "weight_mean": short["set_wt"].mean(),
                "weight_max": short["set_wt"].max(),
            }
        ]
    )
    meta.to_csv(OUT_DIR / "meta_py.csv", index=False)

    pip_lines = os.popen("pip freeze").read().strip().splitlines()
    pip_head = pip_lines[:80]
    session_info = {
        "python_version": os.popen("python --version").read().strip(),
        "platform": os.popen("python -c \"import platform; print(platform.platform())\"")
        .read()
        .strip(),
        "pip_freeze_head": pip_head,
        "pip_freeze_truncated": len(pip_lines) > len(pip_head),
    }
    with open(OUT_DIR / "session_info_py.txt", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(session_info, indent=2))

    truth_path = OUT_DIR / "truth_r.csv"
    if truth_path.exists():
        compare_truth(truth_path, OUT_DIR / "python.csv")


if __name__ == "__main__":
    main()
