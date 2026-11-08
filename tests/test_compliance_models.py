import numpy as np
import pandas as pd

from dailyml.compliance import compliance_by_study_day, compliance_summary, per_day_counts, per_person_compliance
from dailyml.models import bh_adjust, fit_gee, icc, residual_lag1_autocorrelation, rubin_pool, MLResult


def _prompts():
    ts = pd.to_datetime([
        "2021-06-01 09:00", "2021-06-01 13:00", "2021-06-01 19:00",
        "2021-06-02 09:00", "2021-06-02 13:00",
        "2021-06-05 09:00",
    ])
    return pd.DataFrame({
        "id": ["a"] * 5 + ["b"],
        "scheduled_ts": ts,
        "completed": [True, False, True, False, False, True],
    })


def test_per_person_compliance():
    pp = per_person_compliance(_prompts())
    assert pp.loc["a", "scheduled"] == 5 and pp.loc["a", "answered"] == 2
    assert np.isclose(pp.loc["a", "compliance"], 0.4)
    assert pp.loc["b", "compliance"] == 1.0
    s = compliance_summary(pp)
    assert np.isclose(s["mean"], 0.7)
    assert np.isclose(s["pooled"], 3 / 6)


def test_per_day_counts_and_curve():
    d = per_day_counts(_prompts())
    a1 = d[(d.id == "a") & (d.date == "2021-06-01")].iloc[0]
    assert a1.n_scheduled == 3 and a1.n_answered == 2
    curve = compliance_by_study_day(_prompts())
    # day 0: a = 2/3, b = 1 -> mean 0.8333
    assert np.isclose(curve.loc[curve.study_day == 0, "mean"].iloc[0], (2 / 3 + 1) / 2)
    assert curve.loc[curve.study_day == 1, "n_persons"].iloc[0] == 1


def test_icc_recovers_variance_share():
    rng = np.random.default_rng(0)
    n, k = 200, 30
    u = rng.normal(0, np.sqrt(0.3), n)  # between var 0.3
    y = np.repeat(u, k) + rng.normal(0, np.sqrt(0.7), n * k)  # within var 0.7
    df = pd.DataFrame({"id": np.repeat(np.arange(n), k), "y": y})
    r = icc(df, "y")
    assert abs(r["icc"] - 0.3) < 0.05


def test_residual_autocorrelation_skips_gaps():
    rng = np.random.default_rng(2)
    rows = []
    for j in range(50):
        e = np.zeros(60)
        for t in range(1, 60):
            e[t] = 0.5 * e[t - 1] + rng.normal()
        rows.append(pd.DataFrame({"id": j, "date": pd.date_range("2021-06-01", periods=60), "r": e}))
    df = pd.concat(rows, ignore_index=True)
    r = residual_lag1_autocorrelation(df, df["r"])
    assert 0.4 < r["rho1"] < 0.6
    # dropping every other day leaves no consecutive pairs
    sparse = df[df.date.dt.day % 2 == 0]
    assert residual_lag1_autocorrelation(sparse, sparse["r"])["n_pairs"] == 0


def test_gee_logit_recovers_effect():
    rng = np.random.default_rng(4)
    n, k = 150, 40
    x = rng.normal(size=n * k)
    u = np.repeat(rng.normal(0, 0.3, n), k)
    p = 1 / (1 + np.exp(-(-0.2 + 0.8 * x + u)))
    df = pd.DataFrame({"id": np.repeat(np.arange(n), k), "x": x, "y": rng.binomial(1, p)})
    fe = fit_gee(df, "y ~ x", family="binomial").fixed.set_index("term")
    # population-averaged slope is attenuated slightly relative to 0.8
    assert 0.6 < fe.loc["x", "estimate"] < 0.85


def test_rubin_pool_identical_imputations():
    fixed = pd.DataFrame({"term": ["x"], "estimate": [1.0], "se": [0.1], "ci_low": [0.8], "ci_high": [1.2], "p": [0.0]})
    pooled = rubin_pool([MLResult("m", "f", fixed), MLResult("m", "f", fixed)])
    assert np.isclose(pooled.loc[0, "estimate"], 1.0)
    assert np.isclose(pooled.loc[0, "se"], 0.1)


def test_bh_adjust():
    q = bh_adjust(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert np.isclose(q.iloc[0], 0.03) and np.isclose(q.iloc[1], 0.04)
    assert np.isnan(q.iloc[3])
