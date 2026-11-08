"""Experience-sampling (ESM / EMA) compliance and missingness.

Input is a *prompt-level* table: one row per scheduled prompt with at least

* ``id``            person identifier
* ``scheduled_ts``  local time the prompt was delivered (notification)
* ``completed``     bool, True if the prompt was answered before it expired

Compliance is defined as answered / scheduled. Prompts that were never
delivered (e.g. phone off) are invisible in SEMA3 exports and therefore cannot
be counted; reported compliance is an upper bound on "answered / intended".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_person_compliance(prompts: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Scheduled, answered and compliance rate per person."""

    g = prompts.groupby(id_col)["completed"]
    out = pd.DataFrame({"scheduled": g.size(), "answered": g.sum().astype(int)})
    out["compliance"] = out["answered"] / out["scheduled"]
    return out


def per_day_counts(prompts: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Scheduled/answered prompts per person-day (by local scheduled date)."""

    p = prompts.assign(date=prompts["scheduled_ts"].dt.normalize())
    g = p.groupby([id_col, "date"])["completed"]
    out = pd.DataFrame({"n_scheduled": g.size(), "n_answered": g.sum().astype(int)}).reset_index()
    out["day_compliance"] = out["n_answered"] / out["n_scheduled"]
    return out


def compliance_by_study_day(prompts: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Compliance as a function of study day (days since the person's first
    scheduled prompt): mean across persons of each person's daily compliance,
    plus the number of persons contributing and a normal-approximation 95% CI of
    that mean.
    """

    d = per_day_counts(prompts, id_col)
    d["study_day"] = (d["date"] - d.groupby(id_col)["date"].transform("min")).dt.days
    agg = d.groupby("study_day")["day_compliance"].agg(["mean", "std", "count"])
    se = agg["std"] / np.sqrt(agg["count"])
    agg["ci_low"] = (agg["mean"] - 1.96 * se).clip(0, 1)
    agg["ci_high"] = (agg["mean"] + 1.96 * se).clip(0, 1)
    return agg.rename(columns={"count": "n_persons"}).reset_index()


def compliance_summary(per_person: pd.DataFrame) -> dict[str, float]:
    """Distribution of per-person compliance (one number per person)."""

    c = per_person["compliance"]
    return {
        "n_persons": int(c.size),
        "mean": float(c.mean()),
        "sd": float(c.std()),
        "median": float(c.median()),
        "q25": float(c.quantile(0.25)),
        "q75": float(c.quantile(0.75)),
        "share_ge_75pct": float((c >= 0.75).mean()),
        "share_lt_25pct": float((c < 0.25).mean()),
        "pooled": float(per_person["answered"].sum() / per_person["scheduled"].sum()),
    }
