"""Multilevel models for person-day data, built on ``statsmodels``.

What ``statsmodels`` can and cannot do here (version 0.14/0.15):

* ``MixedLM`` fits linear mixed models with random intercepts and random
  slopes per group (one grouping factor). Used for all continuous outcomes.
* ``MixedLM`` has **no residual correlation structures** (no AR(1) like
  ``nlme::lme(correlation = corAR1())``). We therefore (a) measure lag-1
  autocorrelation of the level-1 residuals on consecutive-day pairs, and (b)
  run two sensitivity analyses: a model adding the person-centred lagged
  outcome, and a population-averaged GEE with an autoregressive working
  correlation over calendar time (``cov_struct.Autoregressive(grid=False)``,
  which uses the actual day distance, so gaps are handled).
* There is no frequentist mixed-effects *logistic* model. For binary outcomes
  we use GEE (logit link, exchangeable working correlation, robust SEs;
  population-averaged effects) as the primary model and
  ``BinomialBayesMixedGLM`` (random-intercept logistic model fitted by
  variational Bayes) as a conditional-model check.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MLResult:
    """Tidy summary of a fitted model."""

    name: str
    formula: str
    fixed: pd.DataFrame  # term, estimate, se, ci_low, ci_high, p
    random: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0
    n_groups: int = 0
    converged: bool = True
    notes: list[str] = field(default_factory=list)
    model: object = None  # fitted statsmodels result (not serialised)

    def table(self) -> pd.DataFrame:
        t = self.fixed.copy()
        t.insert(0, "model", self.name)
        t["n_obs"] = self.n_obs
        t["n_persons"] = self.n_groups
        return t


def _fixed_table(params, bse, pvalues, ci, terms) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "term": terms,
            "estimate": params[terms].values,
            "se": bse[terms].values,
            "ci_low": ci.loc[terms, 0].values,
            "ci_high": ci.loc[terms, 1].values,
            "p": pvalues[terms].values,
        }
    )


# ---------------------------------------------------------------------------
# Linear mixed models
# ---------------------------------------------------------------------------


def _drop_missing(df: pd.DataFrame, formula: str, extra: list[str]) -> pd.DataFrame:
    """Complete cases for every variable named in the formula (+ ``extra``)."""

    names = set(extra) | {v for v in df.columns if _mentions(formula, v)}
    return df.dropna(subset=sorted(names))


def _mentions(code: str, var: str) -> bool:
    import re

    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(var)}(?![A-Za-z0-9_])", code) is not None


def _best_fit(model, reml: bool, methods=("bfgs", "lbfgs", "powell")):
    """Fit with several optimisers and keep the converged fit with the highest
    (restricted) log-likelihood. L-BFGS alone can stop at the variance boundary
    (tau = 0) even when the data clearly identify tau > 0, so a single optimiser
    is not trusted."""

    best, notes = None, []
    for m in methods:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                r = model.fit(reml=reml, method=[m])
            except (np.linalg.LinAlgError, ValueError) as e:  # pragma: no cover
                notes.append(f"optimiser {m} failed: {e}")
                continue
        r._fit_warnings = sorted({str(c.message) for c in caught if issubclass(c.category, ConvergenceWarning)})
        if not np.isfinite(r.llf):
            continue
        # prefer converged fits; among equals prefer the higher log-likelihood
        if best is None or (r.converged and not best.converged) or (
            r.converged == best.converged and r.llf > best.llf + 1e-6
        ):
            best = r
    if best is None:
        raise RuntimeError("MixedLM failed with all optimisers: " + "; ".join(notes))
    return best, notes


def fit_mixedlm(
    df: pd.DataFrame,
    formula: str,
    group: str = "id",
    re_formula: str | None = None,
    name: str = "",
    reml: bool = True,
    fallback_to_intercept: bool = True,
) -> MLResult:
    """Fit a linear mixed model and return an :class:`MLResult`.

    ``re_formula`` like ``"~ x_wp"`` adds a random slope (correlated with the
    random intercept). If the random-slope fit does not converge or the slope
    variance collapses to ~0 (boundary), the function refits with a random
    intercept only when ``fallback_to_intercept`` is true and records why in
    ``notes`` -- the data then do not support a random slope.
    """

    extra = [group]
    if re_formula:
        extra += [v for v in df.columns if _mentions(re_formula, v)]
    d = _drop_missing(df, formula, extra)
    notes: list[str] = []
    model = smf.mixedlm(formula, d, groups=d[group], re_formula=re_formula)
    res, fit_notes = _best_fit(model, reml)
    notes += fit_notes
    conv_warn = res._fit_warnings
    converged = bool(res.converged) and not any("hessian" in m.lower() for m in conv_warn)
    random: dict[str, float] = {}
    if res is not None:
        cov = res.cov_re
        for i, a in enumerate(cov.index):
            random[f"var[{a}]"] = float(cov.iloc[i, i])
            for j in range(i):
                random[f"cov[{cov.index[j]},{a}]"] = float(cov.iloc[i, j])
        random["var[residual]"] = float(res.scale)
    slope_var = [v for k, v in random.items() if k.startswith("var[") and "Group" not in k and k != "var[residual]"]
    corr_max = 0.0
    if res is not None and res.cov_re.shape[0] > 1:
        sd = np.sqrt(np.clip(np.diag(res.cov_re), 1e-12, None))
        corr = res.cov_re.to_numpy() / np.outer(sd, sd)
        corr_max = float(np.max(np.abs(corr[np.triu_indices_from(corr, 1)])))
    degenerate = bool(re_formula) and (
        not converged or (slope_var and min(slope_var) < 1e-8) or corr_max > 0.98
    )
    if degenerate and fallback_to_intercept:
        if not converged:
            why = "did not converge"
        elif corr_max > 0.98:
            why = f"singular random-effects covariance (|corr| = {corr_max:.2f})"
        else:
            why = "slope variance at boundary (~0)"
        r0 = fit_mixedlm(df, formula, group, None, name, reml, False)
        r0.notes = [f"random slope {re_formula} dropped: {why}; refit with random intercept only"] + r0.notes
        return r0
    if conv_warn:
        notes += sorted(set(conv_warn))
    terms = list(res.fe_params.index)
    ci = res.conf_int().loc[terms]
    fixed = _fixed_table(res.fe_params, res.bse_fe, res.pvalues[terms], ci, terms)
    return MLResult(
        name=name or formula,
        formula=formula + (f" | re: {re_formula}" if re_formula else ""),
        fixed=fixed,
        random=random,
        n_obs=int(res.nobs),
        n_groups=int(d[group].nunique()),
        converged=converged,
        notes=notes,
        model=res,
    )


def random_slope_lrt(df: pd.DataFrame, formula: str, slope: str, group: str = "id") -> dict[str, float]:
    """REML likelihood-ratio test of a random slope (variance + covariance with
    the intercept) against a random-intercept-only model, on the same rows.

    The null puts the variance on the boundary, so the p value uses the 50:50
    mixture of chi2(1) and chi2(2) (Stram & Lee, 1994; Snijders & Bosker, 2012).
    Fixed effects are identical in both models, so REML likelihoods are comparable.
    """

    d = _drop_missing(df, formula, [group, slope])
    r1, _ = _best_fit(smf.mixedlm(formula, d, groups=d[group], re_formula=f"~ {slope}"), reml=True)
    r0, _ = _best_fit(smf.mixedlm(formula, d, groups=d[group]), reml=True)
    lr = max(0.0, 2 * (r1.llf - r0.llf))
    p = 0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2)
    return {"slope": slope, "lr_stat": lr, "p_mixture": p, "slope_var": float(r1.cov_re.loc[slope, slope]),
            "slope_sd": float(np.sqrt(max(r1.cov_re.loc[slope, slope], 0))), "n_obs": int(r1.nobs)}


def icc(df: pd.DataFrame, outcome: str, group: str = "id") -> dict[str, float]:
    """Intraclass correlation from an intercept-only random-intercept model.

    ICC = tau_00 / (tau_00 + sigma^2): the share of total variance that lies
    between persons. 1 - ICC is the within-person share.
    """

    d = df[[outcome, group]].dropna()
    res, _ = _best_fit(smf.mixedlm(f"{outcome} ~ 1", d, groups=d[group]), reml=True)
    tau = float(res.cov_re.iloc[0, 0])
    sigma2 = float(res.scale)
    return {
        "outcome": outcome,
        "n_obs": int(res.nobs),
        "n_persons": int(d[group].nunique()),
        "between_var": tau,
        "within_var": sigma2,
        "icc": tau / (tau + sigma2) if tau + sigma2 > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def residual_lag1_autocorrelation(
    df: pd.DataFrame, resid: pd.Series, group: str = "id", date: str = "date"
) -> dict[str, float]:
    """Pooled lag-1 correlation of level-1 residuals on *consecutive* calendar
    days of the same person (pairs separated by a gap are not used).
    """

    r = pd.DataFrame({group: df.loc[resid.index, group], date: df.loc[resid.index, date], "r": resid})
    prev = r.assign(**{date: r[date] + pd.Timedelta(days=1)}).rename(columns={"r": "r_prev"})
    pairs = r.merge(prev, on=[group, date]).dropna()
    if len(pairs) < 3:
        return {"rho1": np.nan, "n_pairs": len(pairs), "ci_low": np.nan, "ci_high": np.nan}
    rho = float(np.corrcoef(pairs["r"], pairs["r_prev"])[0, 1])
    z, se = np.arctanh(rho), 1 / np.sqrt(len(pairs) - 3)
    return {
        "rho1": rho,
        "n_pairs": int(len(pairs)),
        "ci_low": float(np.tanh(z - 1.96 * se)),
        "ci_high": float(np.tanh(z + 1.96 * se)),
    }


def fit_gee(
    df: pd.DataFrame,
    formula: str,
    group: str = "id",
    family: str = "gaussian",
    cov: str = "exchangeable",
    time: str | None = None,
    name: str = "",
) -> MLResult:
    """Population-averaged GEE with robust (sandwich) standard errors.

    ``cov="ar"`` uses an autoregressive working correlation whose exponent is
    the absolute difference in ``time`` (e.g. study day), so gaps between days
    lower the assumed correlation rather than being ignored.
    """

    extra = [group] + ([time] if time else [])
    d = _drop_missing(df, formula, extra).sort_values([group] + ([time] if time else []))
    fam = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial()}[family]
    if cov == "ar":
        cs = sm.cov_struct.Autoregressive(grid=False)
        model = smf.gee(formula, group, d, time=d[time].to_numpy(dtype=float), family=fam, cov_struct=cs)
    elif cov == "independence":
        model = smf.gee(formula, group, d, family=fam, cov_struct=sm.cov_struct.Independence())
    else:
        model = smf.gee(formula, group, d, family=fam, cov_struct=sm.cov_struct.Exchangeable())
    notes: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(maxiter=200)
        if not np.all(np.isfinite(res.params)):
            # e.g. clusters with (almost) all-zero binary outcomes make the
            # exchangeable/AR update diverge; independence working correlation
            # with cluster-robust SEs is still valid (less efficient).
            model = smf.gee(formula, group, d, family=fam, cov_struct=sm.cov_struct.Independence())
            res = model.fit(maxiter=200)
            notes.append(f"{cov} working correlation diverged; refit with independence working correlation")
    terms = list(res.params.index)
    fixed = _fixed_table(res.params, res.bse, res.pvalues, res.conf_int(), terms)
    dep = res.model.cov_struct.dep_params
    random = {"working_corr": float(np.atleast_1d(dep)[0]) if dep is not None else np.nan}
    return MLResult(name or formula, formula, fixed, random, int(res.nobs), int(d[group].nunique()), True, notes, res)


def fit_bayes_logit(df: pd.DataFrame, formula: str, group: str = "id", name: str = "") -> MLResult:
    """Random-intercept logistic regression via ``BinomialBayesMixedGLM``
    (variational Bayes, ``fit_vb``). Returns posterior means and +/-1.96 SD
    intervals -- approximate, and to be read as a check on the GEE results.
    """

    d = _drop_missing(df, formula, [group]).copy()
    d["_grp"] = d[group].astype("category").cat.codes
    model = BinomialBayesMixedGLM.from_formula(formula, {"person": "0 + C(_grp)"}, d)
    res = model.fit_vb()
    k = model.k_fep
    terms = list(model.exog_names[:k])
    mean = pd.Series(res.fe_mean, index=terms)
    sd = pd.Series(res.fe_sd, index=terms)
    ci = pd.DataFrame({0: mean - 1.96 * sd, 1: mean + 1.96 * sd})
    p = pd.Series(2 * stats.norm.sf(np.abs(mean / sd)), index=terms)
    fixed = _fixed_table(mean, sd, p, ci, terms)
    random = {"sd[person intercept]": float(np.exp(res.vcp_mean[0]))}
    return MLResult(name or formula, formula, fixed, random, len(d), int(d[group].nunique()), True,
                    ["variational Bayes approximation; 'p' is a normal approximation"], res)


# ---------------------------------------------------------------------------
# Missing data and multiple testing
# ---------------------------------------------------------------------------


def rubin_pool(results: list[MLResult]) -> pd.DataFrame:
    """Pool fixed effects over multiply-imputed fits with Rubin's rules
    (Barnard-Rubin degrees of freedom are approximated by the classic
    Rubin (1987) df)."""

    terms = results[0].fixed["term"].tolist()
    est = np.array([r.fixed.set_index("term").loc[terms, "estimate"].values for r in results])
    se = np.array([r.fixed.set_index("term").loc[terms, "se"].values for r in results])
    m = len(results)
    qbar = est.mean(0)
    ubar = (se**2).mean(0)
    b = est.var(0, ddof=1)
    t = ubar + (1 + 1 / m) * b
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1 + 1 / m) * b / ubar
        df_ = (m - 1) * (1 + 1 / r) ** 2
    df_ = np.where(np.isfinite(df_), df_, 1e6)
    tcrit = stats.t.ppf(0.975, df_)
    sd = np.sqrt(t)
    p = 2 * stats.t.sf(np.abs(qbar / sd), df_)
    return pd.DataFrame(
        {"term": terms, "estimate": qbar, "se": sd, "ci_low": qbar - tcrit * sd, "ci_high": qbar + tcrit * sd,
         "p": p, "fmi": (1 + 1 / m) * b / t}
    )


def bh_adjust(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg false-discovery-rate adjusted p values (q values)."""

    p = pd.Series(p, dtype=float)
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q
