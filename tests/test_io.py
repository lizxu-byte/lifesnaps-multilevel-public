"""Alignment rules of dailyml.io on small synthetic frames (no real data)."""

import numpy as np
import pandas as pd

from dailyml.io import build_person_days, esm_daily, nightly_sleep
from dailyml.prep import add_leads


def _sessions():
    return pd.DataFrame({
        "id": ["a", "a", "a", "a"],
        "startTime": pd.to_datetime(["2021-06-01 23:30", "2021-06-03 01:00", "2021-06-03 14:00", "2021-06-03 22:00"]),
        "endTime": pd.to_datetime(["2021-06-02 07:30", "2021-06-03 08:00", "2021-06-03 15:00", "2021-06-04 06:00"]),
        "dateOfSleep": pd.to_datetime(["2021-06-02", "2021-06-03", "2021-06-03", "2021-06-04"]),
        "minutesAsleep": [450, 380, 55, 470],
        "timeInBed": [480, 420, 60, 480],
        "efficiency": [94, 90, 92, 98],
        "mainSleep": [True, True, False, True],
        "type": ["stages"] * 4,
    })


def test_previous_night_is_assigned_to_wake_day():
    n = nightly_sleep(_sessions()).set_index("date")
    # night 1->2 June belongs to 2 June ("previous night" of 2 June)
    assert n.loc["2021-06-02", "sleep_minutes"] == 450
    # the nap on 3 June (mainSleep False) is ignored
    assert n.loc["2021-06-03", "sleep_minutes"] == 380
    assert n["date_matches_dateOfSleep"].all()
    # clock times measured from the previous midnight: 23:30 -> 23.5, 01:00 -> 25.0
    assert np.isclose(n.loc["2021-06-02", "sleep_onset_h"], 23.5)
    assert np.isclose(n.loc["2021-06-03", "sleep_onset_h"], 25.0)
    assert np.isclose(n.loc["2021-06-02", "sleep_midpoint_h"], (23.5 + 31.5) / 2)


def test_tonight_sleep_is_lead_of_wake_day():
    n = nightly_sleep(_sessions())
    out = add_leads(n, ["sleep_minutes"]).set_index("date")
    # "tonight" for 2 June is the night ending on 3 June
    assert out.loc["2021-06-02", "sleep_minutes_lead1"] == 380


def test_esm_daily_shares_and_missing_days():
    prompts = pd.DataFrame({
        "id": ["a"] * 6,
        "date": pd.to_datetime(["2021-06-02"] * 3 + ["2021-06-03"] * 3),
        "completed": [True, True, False, False, False, False],
        "mood": ["TENSE/ANXIOUS", "HAPPY", np.nan, np.nan, np.nan, np.nan],
        "round": 1,
    })
    e = esm_daily(prompts).set_index("date")
    assert e.loc["2021-06-02", "n_scheduled"] == 3 and e.loc["2021-06-02", "n_answered"] == 2
    assert e.loc["2021-06-02", "esm_stress"] == 0.5
    assert e.loc["2021-06-02", "esm_positive"] == 0.5
    # no answered prompts -> outcome missing, not zero
    assert np.isnan(e.loc["2021-06-03", "esm_stress"])
    assert e.loc["2021-06-03", "n_scheduled"] == 3


def test_build_person_days_inserts_missing_calendar_days():
    fit = pd.DataFrame({"id": ["a", "a"], "date": pd.to_datetime(["2021-06-01", "2021-06-04"]), "steps": [1, 2]})
    wear = pd.DataFrame({"id": ["a"], "date": pd.to_datetime(["2021-06-01"]), "wear_hours": [20]})
    sleep = nightly_sleep(_sessions())
    esm = pd.DataFrame({"id": ["a"], "date": pd.to_datetime(["2021-06-02"]), "esm_stress": [0.5]})
    d = build_person_days(fit, sleep, esm, wear)
    assert d["date"].tolist() == list(pd.date_range("2021-06-01", "2021-06-04"))
    row = d.set_index("date").loc["2021-06-02"]
    assert row["esm_stress"] == 0.5 and row["sleep_minutes"] == 450
