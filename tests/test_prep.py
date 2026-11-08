import numpy as np
import pandas as pd
import pytest

from dailyml.prep import (
    ValidDayRules,
    add_lags,
    add_leads,
    apply_valid_day_rules,
    complete_calendar,
    person_mean_centre,
    study_day,
)
from dailyml.models import fit_mixedlm


def _days(pid, dates, **cols):
    return pd.DataFrame({"id": pid, "date": pd.to_datetime(dates), **cols})


def test_lag_never_bridges_missing_day():
    # day 3 is absent from the table entirely, day 5 present but value missing
    df = _days("a", ["2021-06-01", "2021-06-02", "2021-06-04", "2021-06-05", "2021-06-06"],
               x=[1.0, 2.0, 4.0, np.nan, 6.0])
    out = add_lags(df, ["x"]).set_index("date")["x_lag1"]
    assert out[pd.Timestamp("2021-06-01")] != out[pd.Timestamp("2021-06-01")]  # NaN: no prior day
    assert out[pd.Timestamp("2021-06-02")] == 1.0
    assert np.isnan(out[pd.Timestamp("2021-06-04")])  # 06-03 missing -> must NOT be 2.0
    assert out[pd.Timestamp("2021-06-05")] == 4.0
    assert np.isnan(out[pd.Timestamp("2021-06-06")])  # previous value missing -> NaN


def test_lag_does_not_cross_persons():
    df = pd.concat([_days("a", ["2021-06-01"], x=[1.0]), _days("b", ["2021-06-02"], x=[2.0])])
    out = add_lags(df, ["x"])
    assert out["x_lag1"].isna().all()


def test_lead_is_inverse_of_lag():
    df = _days("a", pd.date_range("2021-06-01", periods=5), x=np.arange(5.0))
    out = add_leads(df, ["x"])
    assert out["x_lead1"].tolist()[:4] == [1.0, 2.0, 3.0, 4.0]
    assert np.isnan(out["x_lead1"].iloc[4])


def test_lag_rejects_duplicate_days():
    df = _days("a", ["2021-06-01", "2021-06-01"], x=[1.0, 2.0])
    with pytest.raises(ValueError):
        add_lags(df, ["x"])


def test_complete_calendar_inserts_gaps():
    df = _days("a", ["2021-06-01", "2021-06-04"], x=[1.0, 4.0])
    out = complete_calendar(df)
    assert len(out) == 4
    assert out["x"].isna().sum() == 2


def test_person_mean_centring_parts():
    df = pd.concat([
        _days("a", pd.date_range("2021-06-01", periods=3), x=[1.0, 2.0, 3.0]),
        _days("b", pd.date_range("2021-06-01", periods=2), x=[10.0, np.nan]),
    ], ignore_index=True)
    out = person_mean_centre(df, ["x"])
    a = out[out.id == "a"]
    assert np.allclose(a["x_pm"], 2.0)
    assert np.allclose(a["x_wp"], [-1, 0, 1])
    # grand mean of person means = (2 + 10) / 2 = 6, each person counted once
    assert np.allclose(a["x_bp"], -4.0)
    assert np.allclose(out.loc[out.id == "b", "x_bp"], 4.0)
    assert np.isnan(out.loc[(out.id == "b"), "x_wp"].iloc[1])
    # within-person deviations average to zero per person
    assert np.allclose(out.groupby("id")["x_wp"].mean(), 0)


def test_centring_scale():
    df = _days("a", pd.date_range("2021-06-01", periods=2), x=[60.0, 120.0])
    out = person_mean_centre(df, ["x"], scale={"x": 60.0})
    assert np.allclose(out["x_wp"], [-0.5, 0.5])


def _simulate(n_persons=80, n_days=40, b_within=-0.5, b_between=1.5, slope_sd=0.0, seed=1):
    """y_ij = u_j + (b_w + v_j) * (x_ij - xbar_j) + b_b * xbar_j + e_ij, with
    person means of x varying a lot so a pooled regression would be confounded."""

    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_persons):
        xbar = rng.normal(0, 2)
        u = rng.normal(0, 1)
        v = rng.normal(0, slope_sd)
        x = xbar + rng.normal(0, 1, n_days)
        xc = x - x.mean()
        y = u + (b_within + v) * xc + b_between * x.mean() + rng.normal(0, 1, n_days)
        rows.append(pd.DataFrame({"id": j, "date": pd.date_range("2021-06-01", periods=n_days), "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_centring_recovers_within_and_between_effects():
    df = person_mean_centre(_simulate(), ["x"])
    res = fit_mixedlm(df, "y ~ x_wp + x_bp", name="sim")
    fe = res.fixed.set_index("term")
    assert fe.loc["x_wp", "ci_low"] < -0.5 < fe.loc["x_wp", "ci_high"]
    assert fe.loc["x_bp", "ci_low"] < 1.5 < fe.loc["x_bp", "ci_high"]
    # a pooled (single-level) regression on raw x conflates the two effects
    import statsmodels.formula.api as smf

    raw = smf.ols("y ~ x", df).fit().params["x"]
    assert 0.0 < raw < 1.5


def test_random_slope_variance_recovered():
    df = person_mean_centre(_simulate(slope_sd=0.4, n_persons=120, seed=3), ["x"])
    res = fit_mixedlm(df, "y ~ x_wp + x_bp", re_formula="~ x_wp", name="rs")
    assert res.converged
    sd = np.sqrt(res.random["var[x_wp]"])
    assert 0.25 < sd < 0.6


def test_random_slope_falls_back_when_absent():
    df = person_mean_centre(_simulate(slope_sd=0.0, seed=5), ["x"])
    res = fit_mixedlm(df, "y ~ x_wp + x_bp", re_formula="~ x_wp", name="rs0")
    # either a tiny slope variance or an explicit fallback note
    if "var[x_wp]" in res.random:
        assert res.random["var[x_wp]"] < 0.05
    else:
        assert any("dropped" in n for n in res.notes)


def test_valid_day_rules():
    df = _days("a", pd.date_range("2021-06-01", periods=3),
               wear_hours=[23, 5, 12], steps=[8000, 3000, 50], rhr=[60, 61, 62],
               sleep_minutes=[420, 60, 1000], sleep_efficiency=[90, 80, 85])
    out = apply_valid_day_rules(df, ValidDayRules())
    assert out["valid_wear_day"].tolist() == [True, False, False]
    assert out["steps"].notna().tolist() == [True, False, False]
    assert out["valid_sleep"].tolist() == [True, False, False]
    assert out["sleep_efficiency"].notna().tolist() == [True, False, False]


def test_study_day():
    df = pd.concat([_days("a", ["2021-06-03", "2021-06-01"]), _days("b", ["2021-11-15"])])
    assert study_day(df).tolist() == [2, 0, 0]
