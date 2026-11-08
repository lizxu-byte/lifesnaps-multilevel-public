"""Run every analysis on LifeSnaps and write aggregated tables/figures to results/.

    python scripts/download_data.py   # once
    python scripts/run_all.py

Nothing written to results/ contains participant IDs or per-day rows.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dailyml import io, plots  # noqa: E402
from dailyml.compliance import (  # noqa: E402
    compliance_by_study_day,
    compliance_summary,
    per_person_compliance,
)
from dailyml.models import (  # noqa: E402
    MLResult,
    bh_adjust,
    fit_bayes_logit,
    fit_gee,
    fit_mixedlm,
    icc,
    random_slope_lrt,
    residual_lag1_autocorrelation,
    rubin_pool,
)
from dailyml.prep import (  # noqa: E402
    DEFAULT_RULES,
    ValidDayRules,
    add_lags,
    add_leads,
    apply_valid_day_rules,
    grand_mean_centre,
    person_mean_centre,
)

warnings.filterwarnings("ignore")
RES = ROOT / "results"
FIG = RES / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
SUMMARY: dict = {}
ALL_FIXED: list[pd.DataFrame] = []
MODEL_NOTES: dict[str, dict] = {}

OUTCOMES = {"esm_stress": "ESM tense/anxious share", "esm_positive": "ESM positive mood share",
            "esm_tired": "ESM tired share"}
LOW_COMPLIANCE = 0.30  # robustness: drop persons answering < 30 % of prompts


def save(df: pd.DataFrame, name: str, float_fmt="%.4g"):
    df.to_csv(RES / name, index=False, float_format=float_fmt)


def record(res: MLResult, family: str = "") -> MLResult:
    t = res.table()
    t["family"] = family
    ALL_FIXED.append(t)
    MODEL_NOTES[res.name] = {"formula": res.formula, "n_obs": res.n_obs, "n_persons": res.n_groups,
                             "converged": res.converged, "random": res.random, "notes": res.notes}
    return res


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------


def build(rules: ValidDayRules = DEFAULT_RULES, sleep_minutes_col: str = "sleep_minutes", stages_only=False):
    prompts = io.load_esm_prompts()
    esm = io.esm_daily(prompts)
    sessions = io.load_sleep_sessions()
    sleep = io.nightly_sleep(sessions)
    if stages_only:
        sleep = sleep[sleep["sleep_type"] == "stages"]
    fitbit = io.load_fitbit_daily()
    wear = io.load_wear_hours()
    d = io.build_person_days(fitbit, sleep, esm, wear)
    d["wear_hours"] = d["wear_hours"].fillna(0)  # no HR record on a calendar day = not worn
    if sleep_minutes_col != "sleep_minutes":
        d["sleep_minutes"] = d[sleep_minutes_col]
    d = apply_valid_day_rules(d, rules)
    # restrict to each person's ESM period (first to last scheduled prompt)
    span = esm.groupby("id")["date"].agg(["min", "max"])
    d = d.merge(span, left_on="id", right_index=True, how="inner")
    d = d[(d["date"] >= d["min"]) & (d["date"] <= d["max"])].drop(columns=["min", "max"])
    d["study_day"] = (d["date"] - d.groupby(["id", "round"])["date"].transform("min")).dt.days
    d["study_week"] = d["study_day"] / 7
    d["sleep_h"] = d["sleep_minutes"] / 60
    d["steps_k"] = d["steps"] / 1000
    d["sleep_eff10"] = d["sleep_efficiency"] / 10
    d = add_lags(d, ["steps_k", "rhr", "wear_hours", "esm_stress", "esm_positive", "esm_tired", "sleep_h"])
    d = add_leads(d, ["sleep_h", "sleep_eff10"])
    d["weekend_night"] = d["weekday"].isin([4, 5]).astype(int)  # Fri/Sat night (next day off)
    d = person_mean_centre(d, ["sleep_h", "sleep_eff10", "sleep_midpoint_h", "steps_k_lag1", "rhr_lag1",
                               "steps_k", "rhr", "esm_stress", "esm_positive", "esm_tired",
                               "esm_stress_lag1", "esm_positive_lag1", "esm_tired_lag1", "wear_hours_lag1",
                               "sleep_h_lag1"])
    parts = io.load_participants()
    parts = parts[parts["id"].isin(d["id"])]
    d = d.merge(parts[["id", "ipip_neuroticism", "stai_mean"]], on="id", how="left")
    for c in ("ipip_neuroticism", "stai_mean"):
        mean, sd = parts[c].mean(), parts[c].std()
        d[f"{c}_z"] = (d[c] - mean) / sd
    return prompts, d, parts, sessions


# ---------------------------------------------------------------------------
# 2. Descriptives + ICC
# ---------------------------------------------------------------------------


def descriptives(prompts, d, parts, sessions):
    pp = per_person_compliance(prompts)
    cs = compliance_summary(pp)
    sleep = io.nightly_sleep(sessions)
    days_esm = d[d["n_mood"] > 0]
    rows = {
        "participants in Fitbit daily CSV": io.load_fitbit_daily()["id"].nunique(),
        "participants with >=1 scheduled mood prompt": prompts["id"].nunique(),
        "participants with >=1 answered mood prompt": prompts.loc[prompts.completed, "id"].nunique(),
        "person-days in ESM periods": len(d),
        "valid wear days (>=10 h HR, >=100 steps)": int(d["valid_wear_day"].sum()),
        "valid main-sleep nights (3-15 h asleep)": int(d["valid_sleep"].sum()),
        "days with >=1 answered mood prompt": len(days_esm),
        "mood prompts scheduled (04:00-24:00)": len(prompts),
        "mood prompts answered": int(prompts["completed"].sum()),
        "pooled ESM compliance": cs["pooled"],
        "per-person compliance: mean": cs["mean"],
        "per-person compliance: SD": cs["sd"],
        "per-person compliance: median": cs["median"],
        "per-person compliance: IQR low": cs["q25"],
        "per-person compliance: IQR high": cs["q75"],
        "share of persons with >=75% compliance": cs["share_ge_75pct"],
        "share of persons with <25% compliance": cs["share_lt_25pct"],
        "median answer latency (min)": float(prompts["latency_min"].median()),
        "main-sleep records ending on Fitbit dateOfSleep (%)": 100 * float(sleep["date_matches_dateOfSleep"].mean()),
        "participants with IPIP": int(parts["ipip_neuroticism"].notna().sum()),
        "participants with >=1 S-STAI": int(parts["stai_mean"].notna().sum()),
    }
    fmt = [str(int(v)) if float(v).is_integer() else f"{v:.4f}" for v in rows.values()]
    desc = pd.DataFrame({"quantity": list(rows), "value": fmt})
    SUMMARY["descriptives"] = rows
    hist = pd.cut(pp["compliance"], np.linspace(0, 1, 11), include_lowest=True).value_counts().sort_index()
    save(pd.DataFrame({"compliance_bin": hist.index.astype(str), "n_persons": hist.values}), "compliance_distribution.csv")

    # per-person summaries of days contributed (distribution only, no IDs)
    per = days_esm.groupby("id").size()
    SUMMARY["esm_days_per_person"] = per.describe().to_dict()

    labels = {"esm_stress": "ESM tense/anxious share", "esm_positive": "ESM positive mood share",
              "esm_tired": "ESM tired share", "sleep_h": "Fitbit sleep duration (h)",
              "sleep_eff10": "Fitbit sleep efficiency", "sleep_midpoint_h": "Fitbit sleep midpoint",
              "steps_k": "Fitbit steps (1000/day)", "rhr": "Fitbit resting HR",
              "fitbit_stress_score": "Fitbit stress score"}
    rows = []
    for v, lab in labels.items():
        r = icc(d, v)
        r["label"] = lab
        rows.append(r)
    icc_t = pd.DataFrame(rows)[["label", "outcome", "n_obs", "n_persons", "between_var", "within_var", "icc"]]
    save(icc_t, "icc.csv")
    plots.icc_plot(icc_t, FIG / "icc.png")
    save(desc, "descriptives.csv")
    return pp


# ---------------------------------------------------------------------------
# 3. Main models: sleep/activity/RHR -> next-day ESM
# ---------------------------------------------------------------------------

PRED_LABEL = {
    "sleep_h": "sleep duration, previous night (per h)",
    "sleep_eff10": "sleep efficiency, previous night (per 10 %-pts)",
    "sleep_midpoint_h": "sleep midpoint, previous night (per h later)",
    "steps_k_lag1": "steps, previous day (per 1000)",
    "rhr_lag1": "resting HR, previous day (per bpm)",
}
MAIN_PREDS = list(PRED_LABEL)
COVARS = "weekend + study_week"


def main_formula(y, preds=MAIN_PREDS, extra=""):
    terms = " + ".join(f"{p}_wp + {p}_bp" for p in preds)
    return f"{y} ~ {terms} + {COVARS}{extra}"


def coef_rows(res: MLResult, outcome_label: str, preds=MAIN_PREDS):
    fe = res.fixed.set_index("term")
    rows = []
    for p in preds:
        for lvl, suf in (("within", "_wp"), ("between", "_bp")):
            if p + suf in fe.index:
                r = fe.loc[p + suf]
                rows.append({"outcome": outcome_label, "predictor": PRED_LABEL.get(p, p), "level": lvl,
                             "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p})
    return rows


def main_models(d):
    coefs, fits = [], {}
    for y, lab in OUTCOMES.items():
        res = fit_mixedlm(d, main_formula(y), re_formula="~ sleep_h_wp", name=f"main:{y}")
        record(res, "confirmatory" if y != "esm_tired" else "secondary")
        fits[y] = res
        coefs += coef_rows(res, lab)
    ct = pd.DataFrame(coefs)
    save(ct, "main_within_between.csv")
    lrt = pd.DataFrame([{"outcome": y, **random_slope_lrt(d, main_formula(y), "sleep_h_wp")} for y in OUTCOMES])
    lrt["fixed_slope"] = [fits[y].fixed.set_index("term").loc["sleep_h_wp", "estimate"] for y in OUTCOMES]
    save(lrt, "random_slope_tests.csv")
    SUMMARY["sd_within_sleep_h"] = float(d.loc[d["esm_stress"].notna(), "sleep_h_wp"].std())
    SUMMARY["sd_between_sleep_h"] = float(d.groupby("id")["sleep_h_pm"].first().std())
    plots.coef_plot(ct[ct.outcome != OUTCOMES["esm_tired"]], FIG / "coef_main.png",
                    "Next-day ESM on previous-night sleep and previous-day activity / RHR")
    plots.coef_plot(ct[ct.outcome == OUTCOMES["esm_tired"]], FIG / "coef_tired.png", "Next-day ESM tiredness")
    return fits


def random_slope_figure(fit: MLResult, d, y):
    res = fit.model
    if "sleep_h_wp" not in res.cov_re.index:
        return
    re = pd.DataFrame(res.random_effects).T
    fe = res.fe_params
    sd = d["sleep_h_wp"].std()
    plots.spaghetti_plot(fe["Intercept"] + fe.get("study_week", 0) * d["study_week"].mean(),
                         fe["sleep_h_wp"], re.iloc[:, 0].to_numpy(), re["sleep_h_wp"].to_numpy(),
                         (-2 * sd, 2 * sd), FIG / f"random_slopes_{y}.png",
                         "previous-night sleep, deviation from own mean (h)",
                         OUTCOMES[y] + " (model-implied)")


def autocorrelation(d, fits):
    rows = []
    for y, fit in fits.items():
        res = fit.model
        resid = pd.Series(res.resid, index=res.model.data.frame.index)
        ac = residual_lag1_autocorrelation(d, resid)
        rows.append({"outcome": y, "model": "main", **ac})
        # sensitivity 1: add person-centred lagged outcome
        f = main_formula(y, extra=f" + {y}_lag1_wp")
        r1 = record(fit_mixedlm(d, f, re_formula="~ sleep_h_wp", name=f"lagged_outcome:{y}"), "sensitivity")
        resid1 = pd.Series(r1.model.resid, index=r1.model.model.data.frame.index)
        rows.append({"outcome": y, "model": "with lagged outcome", **residual_lag1_autocorrelation(d, resid1)})
        # sensitivity 2: GEE with AR working correlation over study day (+ person means as covariates)
        r2 = record(fit_gee(d, main_formula(y), cov="ar", time="study_day", name=f"gee_ar1:{y}"), "sensitivity")
        rows.append({"outcome": y, "model": "GEE AR(1) working corr.", "rho1": r2.random["working_corr"],
                     "n_pairs": np.nan, "ci_low": np.nan, "ci_high": np.nan})
    save(pd.DataFrame(rows), "autocorrelation.csv")


# ---------------------------------------------------------------------------
# 4. Reverse direction: today's ESM -> tonight's sleep
# ---------------------------------------------------------------------------


def reverse_models(d):
    rows = []
    for y, lab in list(OUTCOMES.items())[:2]:
        # previous night's sleep enters as within-person deviation only: its person
        # mean is (almost) the person mean of the outcome itself, which would
        # absorb all between-person variance of tonight's sleep.
        f = (f"sleep_h_lead1 ~ {y}_wp + {y}_bp + sleep_h_wp + steps_k_wp + steps_k_bp"
             f" + weekend_night + study_week")
        res = record(fit_mixedlm(d, f, re_formula=f"~ {y}_wp", name=f"reverse:{y}->sleep"), "confirmatory")
        fe = res.fixed.set_index("term")
        for lvl, suf in (("within", "_wp"), ("between", "_bp")):
            r = fe.loc[y + suf]
            rows.append({"outcome": "tonight's sleep duration (h)", "predictor": lab + " today", "level": lvl,
                         "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p})
        f2 = f.replace("sleep_h_lead1", "sleep_eff10_lead1").replace("sleep_h_wp", "sleep_eff10_wp")
        res2 = record(fit_mixedlm(d, f2, re_formula=None, name=f"reverse:{y}->efficiency"), "secondary")
        fe = res2.fixed.set_index("term")
        for lvl, suf in (("within", "_wp"), ("between", "_bp")):
            r = fe.loc[y + suf]
            rows.append({"outcome": "tonight's sleep efficiency (10 %-pts)", "predictor": lab + " today",
                         "level": lvl, "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p})
    t = pd.DataFrame(rows)
    save(t, "reverse_within_between.csv")
    plots.coef_plot(t, FIG / "coef_reverse.png", "Tonight's sleep on today's ESM reports")


# ---------------------------------------------------------------------------
# 5. Cross-level moderation
# ---------------------------------------------------------------------------


def moderation(d):
    rows = []
    for mod, lab in (("ipip_neuroticism_z", "IPIP neuroticism (z)"), ("stai_mean_z", "mean weekly S-STAI (z)")):
        for y in ("esm_stress", "esm_positive"):
            f = main_formula(y, extra=f" + {mod} + sleep_h_wp:{mod}")
            res = record(fit_mixedlm(d, f, re_formula="~ sleep_h_wp", name=f"moderation:{y}x{mod}"),
                         "confirmatory" if (mod.startswith("ipip") and y == "esm_stress") else "secondary")
            fe = res.fixed.set_index("term")
            for term in ("sleep_h_wp", mod, f"sleep_h_wp:{mod}"):
                r = fe.loc[term]
                rows.append({"outcome": y, "moderator": lab, "term": term, "estimate": r.estimate, "se": r.se,
                             "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p, "n_obs": res.n_obs,
                             "n_persons": res.n_groups,
                             "random_slope_var": res.random.get("var[sleep_h_wp]", np.nan)})
    save(pd.DataFrame(rows), "moderation.csv")


# ---------------------------------------------------------------------------
# 6. ESM compliance model
# ---------------------------------------------------------------------------


def compliance_models(prompts, d):
    # study day counted within person x round (one person took part in both rounds)
    pr = prompts.assign(id=prompts["id"] + "_r" + np.where(prompts["date"] < io.ROUND2_START, "1", "2"))
    curve = compliance_by_study_day(pr)
    save(curve, "compliance_by_study_day.csv")
    plots.compliance_curve(curve, FIG / "compliance_curve.png")

    p = prompts.copy()
    hour = p["scheduled_ts"].dt.hour
    p["slot"] = np.select([hour < 12, hour < 17], ["morning", "afternoon"], "evening")
    p["answered"] = p["completed"].astype(int)
    keep = ["id", "date", "wear_hours_lag1_wp", "wear_hours_lag1_bp", "steps_k_lag1_wp", "steps_k_lag1_bp",
            "sleep_h_wp", "sleep_h_bp", "weekend", "study_week"]
    p = p.merge(d[keep], on=["id", "date"], how="inner")
    p["wear_h_lag1_wp4"] = p["wear_hours_lag1_wp"] / 4  # per 4 h of wear
    p["wear_h_lag1_bp4"] = p["wear_hours_lag1_bp"] / 4
    fA = "answered ~ wear_h_lag1_wp4 + wear_h_lag1_bp4 + weekend + study_week + C(slot, Treatment('morning'))"
    fB = fA + " + steps_k_lag1_wp + steps_k_lag1_bp + sleep_h_wp + sleep_h_bp"
    out = []
    for name, f in (("A: wear time", fA), ("B: + steps, sleep (valid days)", fB)):
        # independence working correlation: the exchangeable update diverges
        # because several persons answered (almost) no prompts
        g = record(fit_gee(p, f, family="binomial", cov="independence", name=f"compliance GEE {name}"),
                   "compliance")
        b = record(fit_bayes_logit(p, f, name=f"compliance BayesMixedGLM {name}"), "compliance")
        for est, res in (("GEE (population-averaged)", g), ("random-intercept logit (VB)", b)):
            t = res.fixed.copy()
            t["odds_ratio"] = np.exp(t["estimate"])
            t["or_ci_low"] = np.exp(t["ci_low"])
            t["or_ci_high"] = np.exp(t["ci_high"])
            t.insert(0, "estimator", est)
            t.insert(0, "model", name)
            t["n_prompts"] = res.n_obs
            t["n_persons"] = res.n_groups
            out.append(t)
    save(pd.concat(out), "compliance_models.csv")
    SUMMARY["compliance_by_week"] = (
        curve[curve["study_day"] < 63].assign(week=curve["study_day"] // 7)
        .groupby("week")["mean"].mean().round(3).to_dict()
    )


# ---------------------------------------------------------------------------
# 7. Robustness
# ---------------------------------------------------------------------------


def robustness(d, pp):
    rows = []

    def add(label, res, y):
        fe = res.fixed.set_index("term")
        for term in ("sleep_h_wp", "sleep_h_bp", "steps_k_lag1_wp", "steps_k_lag1_bp"):
            if term in fe.index:
                r = fe.loc[term]
                rows.append({"variant": label, "outcome": y, "term": term, "estimate": r.estimate, "se": r.se,
                             "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p, "n_obs": res.n_obs,
                             "n_persons": res.n_groups})

    for y in ("esm_stress", "esm_positive"):
        add("main (complete case)", fit_mixedlm(d, main_formula(y), re_formula="~ sleep_h_wp", name="x"), y)
        # sleep duration only (no efficiency / timing / RHR): larger sample
        f = f"{y} ~ sleep_h_wp + sleep_h_bp + steps_k_lag1_wp + steps_k_lag1_bp + {COVARS}"
        add("sleep + steps only", fit_mixedlm(d, f, re_formula="~ sleep_h_wp", name="x"), y)
        # exclude low-compliance persons
        keep_ids = pp.index[pp["compliance"] >= LOW_COMPLIANCE]
        add(f"exclude persons <{LOW_COMPLIANCE:.0%} compliance",
            fit_mixedlm(d[d.id.isin(keep_ids)], main_formula(y), re_formula="~ sleep_h_wp", name="x"), y)
        # only days with >= 2 answered prompts (less noisy daily share)
        add("days with >=2 answered prompts",
            fit_mixedlm(d[d["n_mood"] >= 2], main_formula(y), re_formula="~ sleep_h_wp", name="x"), y)
        # GEE AR(1)
        add("GEE AR(1) working correlation", fit_gee(d, main_formula(y), cov="ar", time="study_day"), y)

    # alternative sleep definitions (rebuild data)
    _, d_tib, _, _ = build(sleep_minutes_col="sleep_tib_minutes")
    _, d_stg, _, _ = build(stages_only=True)
    _, d_strict, _, _ = build(rules=ValidDayRules(min_wear_hours=16, min_sleep_minutes=240, max_sleep_minutes=720))
    for lab, dd in (("sleep = time in bed", d_tib), ("only 'stages' sleep records", d_stg),
                    ("strict rules: wear >=16 h, sleep 4-12 h", d_strict)):
        for y in ("esm_stress", "esm_positive"):
            add(lab, fit_mixedlm(dd, main_formula(y), re_formula="~ sleep_h_wp", name="x"), y)

    # multiple imputation of missing predictors on ESM days
    for y in ("esm_stress", "esm_positive"):
        pooled, n = multiple_imputation(d, y)
        for _, r in pooled.iterrows():
            if r.term in ("sleep_h_wp", "sleep_h_bp", "steps_k_lag1_wp", "steps_k_lag1_bp"):
                rows.append({"variant": "multiple imputation of predictors (m=20)", "outcome": y, "term": r.term,
                             "estimate": r.estimate, "se": r.se, "ci_low": r.ci_low, "ci_high": r.ci_high,
                             "p": r.p, "n_obs": n, "n_persons": d.loc[d[y].notna(), "id"].nunique()})

    # prompt-level outcome definition (binary per answered prompt, GEE logit;
    # coefficients are log-odds, not shares, so they are stored separately)
    save(pd.DataFrame(rows), "robustness.csv")
    prompt_level(d)


def prompt_level(d):
    p = io.load_esm_prompts()
    p = p[p["mood"].notna()].copy()
    p["tense"] = (p["mood"] == "TENSE/ANXIOUS").astype(int)
    p["positive"] = p["mood"].isin(["HAPPY", "RESTED/RELAXED"]).astype(int)
    cols = ["id", "date"] + [c for c in d.columns if c.endswith(("_wp", "_bp"))] + ["weekend", "study_week"]
    p = p.merge(d[cols], on=["id", "date"], how="inner")
    out = []
    for y in ("tense", "positive"):
        f = main_formula(y)
        res = record(fit_gee(p, f, family="binomial", name=f"prompt-level GEE logit:{y}"), "sensitivity")
        t = res.fixed.copy()
        t.insert(0, "outcome", y)
        t["odds_ratio"] = np.exp(t["estimate"])
        t["or_ci_low"], t["or_ci_high"] = np.exp(t["ci_low"]), np.exp(t["ci_high"])
        t["n_prompts"], t["n_persons"] = res.n_obs, res.n_groups
        out.append(t)
    save(pd.concat(out), "prompt_level_logit.csv")


def multiple_imputation(d, y, m=20, seed=11):
    """Impute missing raw predictors on days with an observed ESM outcome.

    Imputation model: statsmodels MICEData (chained equations, predictive mean
    matching) using the other predictors, the outcome, weekend, study week and
    the person's mean of each predictor over observed days (to carry
    between-person information; a single-level approximation to a multilevel
    imputation model). After each imputation the within/between split is
    recomputed and the main model refitted; estimates are pooled with Rubin's
    rules. Assumes data are missing at random given these variables.
    """

    from statsmodels.imputation.mice import MICEData

    raw = ["sleep_h", "sleep_eff10", "sleep_midpoint_h", "steps_k_lag1", "rhr_lag1"]
    sub = d[d[y].notna()].copy()
    aux = sub[["id"]].copy()
    X = sub[raw + [y, "weekend", "study_week"]].copy()
    for c in raw:
        X[f"{c}_pmean"] = sub.groupby("id")[c].transform("mean")
    X = X.fillna({f"{c}_pmean": X[f"{c}_pmean"].mean() for c in raw})
    X = X.reset_index(drop=True)
    imp = MICEData(X, k_pmm=10, rng=np.random.default_rng(seed))
    fits = []
    for _ in range(m):
        imp.update_all(5)
        dd = pd.concat([aux.reset_index(drop=True), sub[["date"]].reset_index(drop=True), imp.data[raw + [y, "weekend", "study_week"]]], axis=1)
        dd = person_mean_centre(dd, raw)
        fits.append(fit_mixedlm(dd, main_formula(y), re_formula=None, name="mi"))
    return rubin_pool(fits), len(sub)


# ---------------------------------------------------------------------------
# 8. Multiple testing + export
# ---------------------------------------------------------------------------


def finish():
    allf = pd.concat(ALL_FIXED, ignore_index=True)
    # BH-FDR over the confirmatory family: within-person terms of the
    # confirmatory models (main stress/positive, reverse duration, IPIP moderation)
    # Family (13 tests): the 5 within-person terms of each main model (stress,
    # positive mood), the within-person ESM term of each reverse model, and the
    # sleep x neuroticism cross-level interaction for stress.
    main_wp = [f"{p}_wp" for p in MAIN_PREDS]
    conf = (
        (allf["model"].isin(["main:esm_stress", "main:esm_positive"]) & allf["term"].isin(main_wp))
        | ((allf["model"] == "reverse:esm_stress->sleep") & (allf["term"] == "esm_stress_wp"))
        | ((allf["model"] == "reverse:esm_positive->sleep") & (allf["term"] == "esm_positive_wp"))
        | ((allf["model"] == "moderation:esm_stressxipip_neuroticism_z")
           & (allf["term"] == "sleep_h_wp:ipip_neuroticism_z"))
    )
    allf["q_bh_confirmatory"] = np.nan
    allf.loc[conf, "q_bh_confirmatory"] = bh_adjust(allf.loc[conf, "p"]).values
    save(allf, "all_fixed_effects.csv")
    (RES / "model_info.json").write_text(json.dumps(MODEL_NOTES, indent=2, default=float))
    (RES / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))


def main():
    prompts, d, parts, sessions = build()
    pp = descriptives(prompts, d, parts, sessions)
    fits = main_models(d)
    for y in ("esm_stress", "esm_positive"):
        random_slope_figure(fits[y], d, y)
    autocorrelation(d, fits)
    reverse_models(d)
    moderation(d)
    compliance_models(prompts, d)
    robustness(d, pp)
    finish()
    print("done; results in", RES)


if __name__ == "__main__":
    main()
