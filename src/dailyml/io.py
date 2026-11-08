"""LifeSnaps loaders: tidy person-day table + participant-level table.

Sources (all inside the git-ignored ``data/`` folder, see
``scripts/download_data.py``):

* ``csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv`` -- Fitbit
  day-level variables (steps, resting HR, Fitbit stress score, ...).
* ``csv_rais_anonymized/hourly_fitbit_sema_df_unprocessed.csv`` -- used only
  for wear time (hours with a heart-rate value).
* ``mongo_rais_anonymized/sema.bson`` -- prompt-level SEMA3 records (scheduled,
  completed and expiry times, MOOD and PLACE answers). The daily CSV only has
  0/1 "was this mood reported at least once today" columns and no record of
  unanswered prompts, so ESM is built from the BSON dump instead.
* ``fitbit.bson`` inside the zip -- streamed once (never extracted, 9.7 GB) to
  pull the 4k ``sleep`` documents, which carry start/end clock times and the
  ``mainSleep`` flag that the CSV does not have.
* ``scored_surveys/*.csv`` -- questionnaire scores.

Alignment rules (person-day table, one row per ``id`` x local calendar ``date``)
---------------------------------------------------------------------------
* **Time zone.** LifeSnaps stores all timestamps in participants' local time
  (paper, "Time Zone Conversion"); daily Fitbit values are already per local
  day. We do not convert anything. Travel across time zones (~3 % of
  participants, per the paper) is not identifiable and is ignored.
* **Night -> day.** A sleep episode belongs to the day on which it *ends*
  (Fitbit's ``dateOfSleep``). Its variables appear in the row of that day as
  ``sleep_*`` and mean "the night before this day". For day *d*:

    - ``sleep_minutes`` on row *d*      = previous night (ended morning of *d*)
    - ``sleep_minutes_lead1`` on row *d* = the night after *d* (ends on *d+1*)

  Only the main sleep (``mainSleep == True``) is used; if several main-sleep
  records end on the same date the longest is kept. Naps are ignored.
* **ESM.** A prompt belongs to the local calendar date of its
  ``SCHEDULED_TS`` (delivery time). Answered = ``COMPLETED_TS`` not null.
  Response contents are attributed to the scheduled date as well (median
  answer latency is ~0.5 h and prompts expire after <= 4 h, so a prompt
  delivered at 20:00 can be answered up to 24:00 at the latest).
  Eight answers with labels outside the app's seven mood options (e.g.
  'JOY', 'FEAR') count as answered but have a missing mood. Prompts
  scheduled before 04:00 (n = 52 of 11,526, odd night-time schedules) are dropped.
* **Activity.** Fitbit daily steps/RHR are for the local calendar day. In the
  models "previous day" activity (``*_lag1``) is used so that predictors
  precede the ESM reports.
"""

from __future__ import annotations

import ast
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
ZIP_NAME = "rais_anonymized.zip"
FITBIT_BSON_MEMBER = "rais_anonymized/mongo_rais_anonymized/fitbit.bson"

MOOD_LEVELS = ["HAPPY", "RESTED/RELAXED", "NEUTRAL", "ALERT", "TIRED", "TENSE/ANXIOUS", "SAD"]
EARLIEST_PROMPT_HOUR = 4
ROUND2_START = pd.Timestamp("2021-10-01")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_bson(path: Path):
    import bson  # from pymongo; imported lazily so tests do not need it

    with open(path, "rb") as fh:
        yield from bson.decode_file_iter(fh)


def _fitbit_datetime(s: pd.Series) -> pd.Series:
    """Fitbit export mixes ``'05/24/21 00:00:00'`` and ISO formats."""

    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    us = pd.to_datetime(s, format="%m/%d/%y %H:%M:%S", errors="coerce")
    return iso.fillna(us)


# ---------------------------------------------------------------------------
# Sleep sessions (streamed from the Fitbit BSON dump)
# ---------------------------------------------------------------------------

_SLEEP_MARKER = b"type\x00\x06\x00\x00\x00sleep\x00"
SLEEP_FIELDS = [
    "logId", "dateOfSleep", "startTime", "endTime", "duration", "minutesAsleep",
    "minutesAwake", "timeInBed", "efficiency", "mainSleep", "type",
]


def extract_sleep_sessions(zip_path: Path | None = None, out_csv: Path | None = None) -> pd.DataFrame:
    """Stream ``fitbit.bson`` out of the zip and keep only ``sleep`` documents.

    The member is decompressed on the fly; only documents whose raw bytes
    contain the ``type == "sleep"`` marker are decoded. The result is cached as
    ``data/derived/sleep_sessions.csv`` (git-ignored).
    """

    import bson

    zip_path = zip_path or DATA / ZIP_NAME
    out_csv = out_csv or DATA / "derived" / "sleep_sessions.csv"
    rows = []
    with zipfile.ZipFile(zip_path) as zf, zf.open(FITBIT_BSON_MEMBER) as fh:
        while True:
            head = fh.read(4)
            if len(head) < 4:
                break
            (length,) = struct.unpack("<i", head)
            raw = head + fh.read(length - 4)
            if _SLEEP_MARKER not in raw:
                continue
            doc = bson.decode(raw)
            if doc.get("type") != "sleep":
                continue
            data = doc["data"]
            row = {"id": str(doc["id"])}
            row.update({k: data.get(k) for k in SLEEP_FIELDS})
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_sleep_sessions(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA / "derived" / "sleep_sessions.csv"
    if not path.exists():
        extract_sleep_sessions(out_csv=path)
    s = pd.read_csv(path)
    for c in ("startTime", "endTime"):
        s[c] = pd.to_datetime(s[c], format="ISO8601")
    s["dateOfSleep"] = pd.to_datetime(s["dateOfSleep"]).dt.normalize()
    s["mainSleep"] = s["mainSleep"].astype(str).str.lower().eq("true")
    return s


def nightly_sleep(sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per (id, date) = main sleep that *ended* on that date.

    Columns: ``sleep_minutes`` (minutes asleep), ``sleep_tib_minutes`` (time in
    bed), ``sleep_efficiency`` (Fitbit, %), ``sleep_onset_h`` (clock hours
    since the previous midnight, i.e. 23:30 -> 23.5, 01:00 -> 25.0),
    ``sleep_midpoint_h`` (same scale) and ``sleep_type`` ("stages"/"classic").
    """

    s = sessions[sessions["mainSleep"]].copy()
    s["date"] = s["endTime"].dt.normalize()
    # Fitbit's dateOfSleep is the wake-up date; keep it as a consistency check.
    s["date_matches_dateOfSleep"] = s["date"].eq(s["dateOfSleep"])
    s = s.sort_values("minutesAsleep", ascending=False).drop_duplicates(["id", "date"])
    prev_midnight = s["date"] - pd.Timedelta(days=1)
    onset = (s["startTime"] - prev_midnight).dt.total_seconds() / 3600
    end = (s["endTime"] - prev_midnight).dt.total_seconds() / 3600
    out = pd.DataFrame({
        "id": s["id"],
        "date": s["date"],
        "sleep_minutes": s["minutesAsleep"].astype(float),
        "sleep_tib_minutes": s["timeInBed"].astype(float),
        "sleep_efficiency": s["efficiency"].astype(float),
        "sleep_onset_h": onset,
        "sleep_midpoint_h": (onset + end) / 2,
        "sleep_type": s["type"],
        "date_matches_dateOfSleep": s["date_matches_dateOfSleep"],
    })
    return out.sort_values(["id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# ESM (SEMA3)
# ---------------------------------------------------------------------------


def load_esm_prompts(path: Path | None = None) -> pd.DataFrame:
    """Prompt-level Context-and-Mood table.

    Columns: ``id``, ``scheduled_ts``, ``completed`` (bool), ``completed_ts``,
    ``mood`` (one of :data:`MOOD_LEVELS`, else NaN), ``place``, ``round``,
    ``latency_min``.
    """

    path = path or DATA / "mongo_rais_anonymized" / "sema.bson"
    recs = []
    for doc in _read_bson(path):
        d = doc["data"]
        if d.get("SURVEY_NAME") != "Context and Mood Survey":
            continue
        recs.append({
            "id": str(doc["user_id"]),
            "scheduled_ts": d.get("SCHEDULED_TS"),
            "completed_ts": d.get("COMPLETED_TS"),
            "expired_ts": d.get("EXPIRED_TS"),
            "mood_raw": d.get("MOOD"),
            "place": d.get("PLACE"),
            "round": 2 if str(d.get("STUDY_NAME", "")).endswith("v2.0") else 1,
        })
    p = pd.DataFrame(recs)
    for c in ("scheduled_ts", "completed_ts", "expired_ts"):
        p[c] = pd.to_datetime(p[c], format="ISO8601")
    p = p[p["scheduled_ts"].dt.hour >= EARLIEST_PROMPT_HOUR].copy()
    p["completed"] = p["completed_ts"].notna()
    p["mood"] = p["mood_raw"].where(p["mood_raw"].isin(MOOD_LEVELS))
    p.loc[~p["completed"], ["mood", "place"]] = np.nan
    p["latency_min"] = (p["completed_ts"] - p["scheduled_ts"]).dt.total_seconds() / 60
    p["date"] = p["scheduled_ts"].dt.normalize()
    return p.sort_values(["id", "scheduled_ts"]).reset_index(drop=True)


def esm_daily(prompts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prompts to person-days.

    ``esm_stress``  = share of answered prompts that day reporting TENSE/ANXIOUS
    ``esm_positive``= share reporting HAPPY or RESTED/RELAXED
    ``esm_tired``   = share reporting TIRED
    Shares are NaN on days with no answered prompt (not 0).
    """

    p = prompts.copy()
    p["tense"] = (p["mood"] == "TENSE/ANXIOUS").astype(float)
    p["positive"] = p["mood"].isin(["HAPPY", "RESTED/RELAXED"]).astype(float)
    p["tired"] = (p["mood"] == "TIRED").astype(float)
    p["has_mood"] = p["mood"].notna()
    for c in ("tense", "positive", "tired"):
        p.loc[~p["has_mood"], c] = np.nan
    g = p.groupby(["id", "date"])
    out = pd.DataFrame({
        "n_scheduled": g.size(),
        "n_answered": g["completed"].sum().astype(int),
        "n_mood": g["has_mood"].sum().astype(int),
        "esm_stress": g["tense"].mean(),
        "esm_positive": g["positive"].mean(),
        "esm_tired": g["tired"].mean(),
    }).reset_index()
    return out


# ---------------------------------------------------------------------------
# Fitbit day level
# ---------------------------------------------------------------------------


def load_fitbit_daily(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv"
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    keep = {
        "steps": "steps",
        "resting_hr": "rhr",
        "stress_score": "fitbit_stress_score",
        "very_active_minutes": "very_active_min",
        "moderately_active_minutes": "moderately_active_min",
        "sedentary_minutes": "sedentary_min",
        "minutesAsleep": "csv_minutes_asleep",
    }
    out = d[["id", "date"] + list(keep)].rename(columns=keep)
    # Fitbit stress score: 0 is documented as an out-of-range artefact (paper).
    out.loc[out["fitbit_stress_score"] <= 0, "fitbit_stress_score"] = np.nan
    if out.duplicated(["id", "date"]).any():
        out = out.groupby(["id", "date"], as_index=False).first()
    return out


def load_wear_hours(path: Path | None = None) -> pd.DataFrame:
    """Hours (0-24) per person-day with at least one heart-rate value."""

    path = path or DATA / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv"
    h = pd.read_csv(path, usecols=["id", "date", "hour", "bpm"])
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    # count distinct clock hours (the export has a few duplicated hours, e.g. DST)
    w = h[h["bpm"].notna()].groupby(["id", "date"])["hour"].nunique()
    return w.rename("wear_hours").reset_index()


# ---------------------------------------------------------------------------
# Questionnaires (participant level)
# ---------------------------------------------------------------------------


def load_participants(folder: Path | None = None, daily_csv: Path | None = None) -> pd.DataFrame:
    """One row per participant.

    * IPIP Big Five (entry; sum scores): ``ipip_neuroticism`` is derived as the
      negated *Emotional Stability* score so that higher = more neurotic.
    * S-STAI (weekly, *state* anxiety; trait STAI was not administered): the
      person mean over weeks, ``stai_mean``, is used as a trait-like summary.
    * PANAS (weekly): person means of positive/negative affect.
    * gender and age group (from the daily CSV; only two age groups released).
    """

    folder = folder or DATA / "scored_surveys"
    pers = pd.read_csv(folder / "personality.csv")
    pers = pers.rename(columns={"user_id": "id"})
    stai = pd.read_csv(folder / "stai.csv").rename(columns={"user_id": "id"})
    panas = pd.read_csv(folder / "panas.csv").rename(columns={"user_id": "id"})
    out = pd.DataFrame({"id": pd.unique(pd.concat([pers["id"], stai["id"], panas["id"]]))})

    # IPIP: keep the first (entry) response per person
    if "submitdate" in pers:
        pers = pers.sort_values("submitdate")
    pers = pers.drop_duplicates("id")
    es = [c for c in pers.columns if "stability" in c.lower() and pd.api.types.is_numeric_dtype(pers[c])]
    if es:
        out = out.merge(pers[["id", es[0]]].rename(columns={es[0]: "ipip_emotional_stability"}), on="id", how="left")
        out["ipip_neuroticism"] = -out["ipip_emotional_stability"]

    stai_col = [c for c in stai.columns if c.lower() in ("stai_stress", "stai", "stai_score")]
    if stai_col:
        s = stai.groupby("id")[stai_col[0]].agg(["mean", "size"]).rename(columns={"mean": "stai_mean", "size": "stai_n"})
        out = out.merge(s.reset_index(), on="id", how="left")
    for col, name in (("positive_affect_score", "panas_pa_mean"), ("negative_affect_score", "panas_na_mean")):
        if col in panas:
            out = out.merge(panas.groupby("id")[col].mean().rename(name).reset_index(), on="id", how="left")

    daily_csv = daily_csv or DATA / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv"
    demo = pd.read_csv(daily_csv, usecols=["id", "gender", "age"]).dropna().drop_duplicates("id")
    return out.merge(demo, on="id", how="outer")


# ---------------------------------------------------------------------------
# Person-day table
# ---------------------------------------------------------------------------


def build_person_days(
    fitbit: pd.DataFrame, sleep: pd.DataFrame, esm: pd.DataFrame, wear: pd.DataFrame
) -> pd.DataFrame:
    """Outer-join the day-level sources on (id, date).

    The calendar for each person spans from their first to their last day with
    *any* record (Fitbit or ESM); missing days inside that span are inserted as
    empty rows so that lags can be computed honestly.
    """

    from .prep import complete_calendar

    d = fitbit.merge(wear, on=["id", "date"], how="outer")
    d = d.merge(sleep, on=["id", "date"], how="outer")
    d = d.merge(esm, on=["id", "date"], how="outer")
    d = d.sort_values(["id", "date"]).reset_index(drop=True)
    d = complete_calendar(d)
    # Round 1 ran May-Jul 2021, round 2 Nov 2021-Jan 2022 (paper); one person
    # took part in both, so study time is always counted within round.
    d["round"] = np.where(d["date"] < ROUND2_START, 1, 2)
    d["weekday"] = d["date"].dt.dayofweek
    d["weekend"] = (d["weekday"] >= 5).astype(int)
    return d
