"""Preparation of the person-day table for multilevel modelling.

Everything in this module is data-source agnostic: it works on any long table
with one row per (person, calendar day). The LifeSnaps-specific parts live in
:mod:`dailyml.io`.

Conventions
-----------
* ``id`` column identifies the person, ``date`` is a ``datetime64`` calendar
  day (normalised to midnight). Both names are configurable.
* For a time-varying predictor ``x`` the decomposition is::

      x_pm  = person mean of x over that person's *non-missing* days
      x_wp  = x - x_pm                         (within-person deviation)
      x_bp  = x_pm - mean over persons of x_pm (between-person, grand-mean centred)

  ``x_wp`` and ``x_bp`` are orthogonal by construction and enter the model
  together (Curran & Bauer, 2011; Enders & Tofighi, 2007).
* Lags are calendar lags: the lag-1 value for day *d* is the value observed on
  day *d - 1* for the same person. If day *d - 1* is absent or missing, the lag
  is missing; it never silently takes the value of the last observed day.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Valid-day rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidDayRules:
    """Thresholds for declaring wearable data on a day usable.

    Attributes
    ----------
    min_wear_hours:
        Minimum number of clock hours (0-23) that contain at least one heart
        rate sample for the day to count as a valid *wear day*. 10 h is the
        conventional accelerometry threshold (Troiano et al., 2008); here wear is
        approximated by optical heart-rate presence because LifeSnaps does not
        export a Fitbit wear flag.
    min_sleep_minutes, max_sleep_minutes:
        Plausible range for main-sleep minutes asleep. Nights outside the range
        (e.g. a 40-min nap logged as main sleep, or a 20-h artefact) are set to
        missing.
    min_steps:
        Days with fewer steps are treated as non-wear for activity variables
        (a worn tracker essentially never records < 100 steps in a day).
    """

    min_wear_hours: int = 10
    min_sleep_minutes: float = 180.0
    max_sleep_minutes: float = 900.0
    min_steps: float = 100.0


DEFAULT_RULES = ValidDayRules()


def apply_valid_day_rules(
    df: pd.DataFrame,
    rules: ValidDayRules = DEFAULT_RULES,
    wear_col: str = "wear_hours",
    sleep_col: str = "sleep_minutes",
    steps_col: str = "steps",
    daytime_cols: Iterable[str] = ("steps", "rhr"),
    sleep_cols: Iterable[str] = ("sleep_minutes", "sleep_efficiency", "sleep_midpoint_h"),
) -> pd.DataFrame:
    """Return a copy with invalid wearable values set to ``NaN``.

    Adds boolean columns ``valid_wear_day`` and ``valid_sleep``. Daytime
    variables are blanked on non-valid wear days; sleep variables are blanked
    when the sleep duration is implausible. ESM columns are never touched.
    """

    out = df.copy()
    wear_ok = out[wear_col].fillna(0) >= rules.min_wear_hours
    if steps_col in out:
        wear_ok &= out[steps_col].fillna(0) >= rules.min_steps
    out["valid_wear_day"] = wear_ok
    for c in daytime_cols:
        if c in out:
            out.loc[~wear_ok, c] = np.nan

    s = out[sleep_col]
    sleep_ok = s.between(rules.min_sleep_minutes, rules.max_sleep_minutes)
    out["valid_sleep"] = sleep_ok
    for c in sleep_cols:
        if c in out:
            out.loc[~sleep_ok, c] = np.nan
    return out


# ---------------------------------------------------------------------------
# Calendar completion and lags
# ---------------------------------------------------------------------------


def complete_calendar(df: pd.DataFrame, id_col: str = "id", date_col: str = "date") -> pd.DataFrame:
    """Insert empty rows so every person has every calendar day between their
    first and last row. Existing values are untouched; new rows are all-NaN.
    """

    pieces = []
    for pid, g in df.groupby(id_col, sort=False):
        full = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
        g = g.set_index(date_col).reindex(full)
        g.index.name = date_col
        g[id_col] = pid
        pieces.append(g.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    return out[df.columns.tolist()]


def add_lags(
    df: pd.DataFrame,
    cols: Iterable[str],
    lag: int = 1,
    id_col: str = "id",
    date_col: str = "date",
    suffix: str | None = None,
) -> pd.DataFrame:
    """Add calendar-lagged copies of ``cols``.

    ``x_lag1`` on day *d* equals ``x`` on day *d - lag* of the same person, or
    NaN if that day is not in the table or ``x`` is missing on it. Works on
    tables with gaps (no need to call :func:`complete_calendar` first) because
    the lag is a merge on the shifted date, not a positional shift.
    """

    cols = list(cols)
    suffix = suffix or f"_lag{lag}"
    src = df[[id_col, date_col] + cols].copy()
    src[date_col] = src[date_col] + pd.Timedelta(days=lag)
    src = src.rename(columns={c: c + suffix for c in cols})
    if src.duplicated([id_col, date_col]).any():
        raise ValueError("duplicate (id, date) rows; lags are ill-defined")
    return df.merge(src, on=[id_col, date_col], how="left")


def add_leads(df, cols, lead: int = 1, id_col: str = "id", date_col: str = "date"):
    """Like :func:`add_lags` but forward in time (``x_lead1`` = value on d + 1)."""

    return add_lags(df, cols, lag=-lead, id_col=id_col, date_col=date_col, suffix=f"_lead{lead}")


# ---------------------------------------------------------------------------
# Within/between decomposition
# ---------------------------------------------------------------------------


def person_mean_centre(
    df: pd.DataFrame,
    cols: Iterable[str],
    id_col: str = "id",
    scale: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Split each column into person mean, within-person and between-person parts.

    For every ``c`` in ``cols`` adds ``c_pm``, ``c_wp`` and ``c_bp`` (see module
    docstring). The grand mean used for ``c_bp`` is the unweighted mean of the
    person means (each person counts once), so the between-person coefficient is
    not dominated by people with many days.

    ``scale`` optionally divides a variable by a constant first (e.g. minutes to
    hours) so that coefficients are per meaningful unit.
    """

    out = df.copy()
    scale = scale or {}
    for c in cols:
        x = out[c] / scale.get(c, 1.0)
        pm = x.groupby(out[id_col]).transform("mean")
        gm = pm.groupby(out[id_col]).first().mean()
        out[f"{c}_pm"] = pm
        out[f"{c}_wp"] = x - pm
        out[f"{c}_bp"] = pm - gm
    return out


def grand_mean_centre(df: pd.DataFrame, cols: Iterable[str], id_col: str = "id") -> pd.DataFrame:
    """Centre person-level (time-invariant) variables at the mean over persons."""

    out = df.copy()
    for c in cols:
        per_person = out.groupby(id_col)[c].first()
        out[f"{c}_c"] = out[c] - per_person.mean()
    return out


def study_day(df: pd.DataFrame, id_col: str = "id", date_col: str = "date") -> pd.Series:
    """Days since the person's first row (0-based)."""

    first = df.groupby(id_col)[date_col].transform("min")
    return (df[date_col] - first).dt.days
