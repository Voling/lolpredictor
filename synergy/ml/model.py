import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import (
    CONTROL_COLUMNS,
    CROSS_COLUMNS,
    CROSS_TERMS,
    DIFF_COLUMNS,
    HISTORY_COLUMNS,
    PHI_COLUMNS,
    STYLE_NAMES,
    STYLE_SUM_COLUMNS,
    build_team_dataset,
)

RIDGE_ALPHAS = np.logspace(1, 6, 24)
MIN_SYNERGY_SPREAD = 1e-3
MIN_SYNERGY_GAIN_SIGMA = 2.0
ADVANTAGE_NULLS = 20
SCORE_SPAN = 6.0
OBSERVED_LINEUP = 3
PROBABILITY_FLOOR = 1e-4


@dataclass
class TrainingReport:
    matches: int = 0
    pair_rows: int = 0
    pairs: int = 0
    baseline_auc: float = 0.0
    baseline_log_loss: float = 0.0
    auc: float = 0.0
    log_loss: float = 0.0
    brier: float = 0.0
    ridge_alpha: float = 0.0
    synergy_std: float = 0.0
    synergy_mean: float = 0.0
    synergy_gain: float = 0.0
    synergy_gain_sigma: float = 0.0
    advantage_matches: int = 0
    advantage_sd: float = 0.0
    advantage_base_r2: float = 0.0
    advantage_full_r2: float = 0.0
    advantage_gain: float = 0.0
    advantage_null_mean: float = 0.0
    advantage_null_sd: float = 0.0
    advantage_nulls_above: int = 0
    advantage_gold_sd: float = 0.0
    style_only_auc: float = 0.0
    observed_matches: int = 0
    observed_baseline_auc: float = 0.0
    observed_auc: float = 0.0
    observed_style_only_auc: float = 0.0
    top_terms: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "matches": self.matches,
            "pair_rows": self.pair_rows,
            "pairs": self.pairs,
            "baseline_auc": round(self.baseline_auc, 4),
            "baseline_log_loss": round(self.baseline_log_loss, 4),
            "auc": round(self.auc, 4),
            "log_loss": round(self.log_loss, 4),
            "brier": round(self.brier, 4),
            "style_only_auc": round(self.style_only_auc, 4),
            "observed_matches": self.observed_matches,
            "observed_baseline_auc": round(self.observed_baseline_auc, 4),
            "observed_auc": round(self.observed_auc, 4),
            "observed_style_only_auc": round(self.observed_style_only_auc, 4),
            "ridge_alpha": round(float(self.ridge_alpha), 3),
            "synergy_std": round(self.synergy_std, 4),
            "synergy_mean": round(self.synergy_mean, 5),
            "synergy_gain": round(self.synergy_gain, 4),
            "synergy_gain_sigma": round(self.synergy_gain_sigma, 2),
            "advantage_matches": self.advantage_matches,
            "advantage_sd": round(self.advantage_sd, 1),
            "advantage_base_r2": round(self.advantage_base_r2, 6),
            "advantage_full_r2": round(self.advantage_full_r2, 6),
            "advantage_gain": round(self.advantage_gain, 6),
            "advantage_null_mean": round(self.advantage_null_mean, 6),
            "advantage_null_sd": round(self.advantage_null_sd, 6),
            "advantage_nulls_above": self.advantage_nulls_above,
            "advantage_gold_sd": round(self.advantage_gold_sd, 1),
            "top_terms": self.top_terms,
        }


def _baseline() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=2000))])


def _residual() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=RIDGE_ALPHAS))])


class SynergyModel:
    def __init__(self):
        self.baseline: Pipeline | None = None
        self.residual: Pipeline | None = None
        self.advantage: Pipeline | None = None
        self.advantage_weights: pd.Series = pd.Series(dtype=float)
        self.weights: pd.Series = pd.Series(dtype=float)
        self.quantiles: np.ndarray = np.array([])
        self.report = TrainingReport()

    def _raw_weights(self, pipeline: Pipeline, columns: list[str]) -> pd.Series:
        scale = pipeline.named_steps["scale"].scale_
        coefficients = pipeline.named_steps["model"].coef_
        coefficients = coefficients[0] if coefficients.ndim > 1 else coefficients
        return pd.Series(coefficients / np.where(scale == 0, 1.0, scale), index=columns)

    def fit(
        self,
        features: pd.DataFrame,
        controls: pd.DataFrame,
        folds: int = 5,
        advantage: pd.Series | None = None,
    ) -> TrainingReport:
        X, y, observed = build_team_dataset(features, controls)
        report = TrainingReport(
            matches=len(X), pair_rows=len(features), pairs=int(features["pair_key"].nunique())
        )
        phi = X[PHI_COLUMNS]
        control = X[CONTROL_COLUMNS]
        if len(X) >= 100:
            splitter = KFold(n_splits=min(folds, len(X) // 20), shuffle=True, random_state=42)
            baseline_out = np.zeros(len(X))
            style_out = np.zeros(len(X))
            combined_out = np.zeros(len(X))
            for train_index, test_index in splitter.split(X):
                baseline = _baseline().fit(control.iloc[train_index], y[train_index])
                baseline_out[test_index] = baseline.predict_proba(control.iloc[test_index])[:, 1]
                style = _baseline().fit(X[STYLE_SUM_COLUMNS].iloc[train_index], y[train_index])
                style_out[test_index] = style.predict_proba(
                    X[STYLE_SUM_COLUMNS].iloc[test_index]
                )[:, 1]
                in_fold = baseline.predict_proba(control.iloc[train_index])[:, 1]
                residual = _residual().fit(phi.iloc[train_index], y[train_index] - in_fold)
                combined_out[test_index] = baseline_out[test_index] + residual.predict(
                    phi.iloc[test_index]
                )
            combined_out = np.clip(combined_out, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
            report.baseline_auc = float(roc_auc_score(y, baseline_out))
            report.baseline_log_loss = float(log_loss(y, baseline_out))
            report.auc = float(roc_auc_score(y, combined_out))
            report.log_loss = float(log_loss(y, combined_out))
            report.brier = float(brier_score_loss(y, combined_out))
            report.style_only_auc = float(roc_auc_score(y, style_out))
            stratum = observed >= OBSERVED_LINEUP
            report.observed_matches = int(stratum.sum())
            if report.observed_matches >= 100 and len(set(y[stratum])) == 2:
                report.observed_baseline_auc = float(roc_auc_score(y[stratum], baseline_out[stratum]))
                report.observed_auc = float(roc_auc_score(y[stratum], combined_out[stratum]))
                report.observed_style_only_auc = float(roc_auc_score(y[stratum], style_out[stratum]))
            report.synergy_gain = report.auc - report.baseline_auc
            report.synergy_gain_sigma = report.synergy_gain / (0.5 / np.sqrt(max(len(y), 4) / 4.0))

        target = None if advantage is None else advantage.reindex(X.index)
        if target is not None and target.notna().sum() >= 2000:
            self._fit_advantage(report, X, phi, control, target, folds)

        self.baseline = _baseline().fit(control, y)
        expected = self.baseline.predict_proba(control)[:, 1]
        self.residual = _residual().fit(phi, y - expected)
        self.weights = self._raw_weights(self.residual, PHI_COLUMNS)
        report.ridge_alpha = float(self.residual.named_steps["model"].alpha_)

        synergy = self.synergy(features)
        report.synergy_std = float(np.std(synergy))
        report.advantage_gold_sd = round(float(np.std(synergy) * report.advantage_sd), 1)
        report.synergy_mean = float(np.mean(synergy))
        self.quantiles = np.quantile(synergy, np.linspace(0, 1, 1001))
        ordered = self.weights.reindex(self.weights.abs().sort_values(ascending=False).index)
        report.top_terms = [
            {"term": name, "weight": round(float(value), 5)} for name, value in ordered.head(12).items()
        ]
        self.report = report
        return report

    def _fit_advantage(self, report, X, phi, control, target, folds) -> None:
        keep = target.notna().to_numpy()
        gold = target.to_numpy(dtype=float)[keep]
        report.advantage_matches = int(keep.sum())
        report.advantage_sd = float(np.std(gold))
        scaled = (gold - gold.mean()) / max(float(np.std(gold)), 1e-9)
        block, side = phi[keep], control[keep]
        splitter = KFold(n_splits=min(folds, max(2, len(gold) // 20)), shuffle=True, random_state=42)

        def held_out(matrix, values):
            out = np.zeros(len(values))
            for fit, test in splitter.split(matrix):
                model = _residual().fit(matrix.iloc[fit], values[fit])
                out[test] = model.predict(matrix.iloc[test])
            total = ((values - values.mean()) ** 2).sum()
            return 1.0 - ((values - out) ** 2).sum() / total, out

        base_r2, base_out = held_out(side, scaled)
        joined = pd.concat([side, block], axis=1)
        full_r2, _ = held_out(joined, scaled)
        report.advantage_base_r2 = float(base_r2)
        report.advantage_full_r2 = float(full_r2)
        report.advantage_gain = float(full_r2 - base_r2)

        rng = np.random.default_rng(0)
        draws = []
        for _ in range(ADVANTAGE_NULLS):
            spun = block.to_numpy(dtype=float)[rng.permutation(len(block))]
            fake = pd.concat(
                [side, pd.DataFrame(spun, index=side.index, columns=block.columns)], axis=1
            )
            draws.append(held_out(fake, scaled)[0] - base_r2)
        draws = np.asarray(draws)
        report.advantage_null_mean = float(draws.mean())
        report.advantage_null_sd = float(draws.std(ddof=1))
        report.advantage_nulls_above = int((draws >= report.advantage_gain).sum())

        self.advantage = _residual().fit(joined, scaled - base_out)
        self.advantage_weights = self._raw_weights(self.advantage, list(joined.columns))

    def synergy(self, phi: pd.DataFrame) -> np.ndarray:
        weights = self.pair_weights
        if weights.empty:
            raise RuntimeError("model not trained")
        matrix = phi.reindex(columns=list(weights.index)).fillna(0.0).to_numpy(dtype=float)
        return matrix @ weights.to_numpy()

    @property
    def pair_weights(self) -> pd.Series:
        if not getattr(self, "advantage_weights", pd.Series(dtype=float)).empty:
            return self.advantage_weights.reindex(PHI_COLUMNS).dropna()
        return self.weights

    @property
    def informative(self) -> bool:
        if float(self.report.synergy_std) < MIN_SYNERGY_SPREAD:
            return False
        if self.report.advantage_matches:
            return self.report.advantage_gain > 0.0 and self.report.advantage_nulls_above == 0
        return float(self.report.synergy_gain_sigma) >= MIN_SYNERGY_GAIN_SIGMA

    def score(self, synergy: np.ndarray | float) -> np.ndarray:
        values = np.atleast_1d(np.asarray(synergy, dtype=float))
        spread = max(float(self.report.synergy_std), MIN_SYNERGY_SPREAD)
        centred = values - float(self.report.synergy_mean)
        return np.clip(50.0 + 50.0 * np.tanh(centred / (SCORE_SPAN * spread)), 0.0, 100.0)

    def contributions(self, phi: pd.DataFrame) -> pd.Series:
        weights = self.pair_weights
        row = phi.reindex(columns=list(weights.index)).fillna(0.0).iloc[0]
        return row * weights

    def explain(self, phi: pd.DataFrame, top: int = 5) -> list[dict]:
        contributions = self.contributions(phi)
        by_axis: dict[str, float] = {name: 0.0 for name in STYLE_NAMES}
        by_axis["shared_history"] = 0.0
        for column, (first, second) in zip(CROSS_COLUMNS, CROSS_TERMS):
            if column not in contributions:
                continue
            value = float(contributions[column])
            by_axis[first] += value / 2.0
            by_axis[second] += value / 2.0
        for column, name in zip(DIFF_COLUMNS, STYLE_NAMES):
            if column in contributions:
                by_axis[name] += float(contributions[column])
        for column in HISTORY_COLUMNS:
            if column in contributions:
                by_axis["shared_history"] += float(contributions[column])
        ranked = sorted(by_axis.items(), key=lambda item: abs(item[1]), reverse=True)
        return [{"axis": axis, "impact": float(value)} for axis, value in ranked[:top]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "SynergyModel":
        with open(path, "rb") as handle:
            return pickle.load(handle)
