import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
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
    build_team_dataset,
)

RIDGE_ALPHAS = np.logspace(1, 6, 24)
MIN_SYNERGY_SPREAD = 1e-3
ADVANTAGE_NULLS = 20
MIN_ADVANTAGE_MATCHES = 200
SCORE_SPAN = 6.0


@dataclass
class TrainingReport:
    matches: int = 0
    pair_rows: int = 0
    pairs: int = 0
    ridge_alpha: float = 0.0
    synergy_std: float = 0.0
    synergy_mean: float = 0.0
    advantage_matches: int = 0
    advantage_sd: float = 0.0
    advantage_base_r2: float = 0.0
    advantage_full_r2: float = 0.0
    advantage_gain: float = 0.0
    advantage_null_mean: float = 0.0
    advantage_null_sd: float = 0.0
    advantage_nulls_above: int = 0
    advantage_gold_sd: float = 0.0
    phi_columns: list[str] = field(default_factory=list)
    top_terms: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "matches": self.matches,
            "pair_rows": self.pair_rows,
            "pairs": self.pairs,
            "target": "gold plus objectives at 15, blue minus red",
            "ridge_alpha": round(float(self.ridge_alpha), 3),
            "synergy_std": round(self.synergy_std, 4),
            "synergy_mean": round(self.synergy_mean, 5),
            "advantage_matches": self.advantage_matches,
            "advantage_sd": round(self.advantage_sd, 1),
            "advantage_base_r2": round(self.advantage_base_r2, 6),
            "advantage_full_r2": round(self.advantage_full_r2, 6),
            "advantage_gain": round(self.advantage_gain, 6),
            "advantage_null_mean": round(self.advantage_null_mean, 6),
            "advantage_null_sd": round(self.advantage_null_sd, 6),
            "advantage_nulls_above": self.advantage_nulls_above,
            "advantage_gold_sd": round(self.advantage_gold_sd, 1),
            "phi_columns": len(self.phi_columns),
            "top_terms": self.top_terms,
        }


def _residual() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=RIDGE_ALPHAS))])


class SynergyModel:
    def __init__(self):
        self.advantage: Pipeline | None = None
        self.advantage_weights: pd.Series = pd.Series(dtype=float)
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
        X, _, _ = build_team_dataset(features, controls)
        report = TrainingReport(
            matches=len(X), pair_rows=len(features), pairs=int(features["pair_key"].nunique())
        )
        report.phi_columns = list(PHI_COLUMNS)
        phi = X[PHI_COLUMNS]
        control = X[CONTROL_COLUMNS]
        if advantage is None:
            raise ValueError("the pair model trains on advantage at 15 and none was supplied")
        target = advantage.reindex(X.index)
        if target.notna().sum() < MIN_ADVANTAGE_MATCHES:
            raise ValueError(f"only {int(target.notna().sum())} matches carry advantage at 15")
        self._fit_advantage(report, X, phi, control, target, folds)
        report.ridge_alpha = float(self.advantage.named_steps["model"].alpha_)

        synergy = self.synergy(features)
        report.synergy_std = float(np.std(synergy))
        report.advantage_gold_sd = round(float(np.std(synergy) * report.advantage_sd), 1)
        report.synergy_mean = float(np.mean(synergy))
        self.quantiles = np.quantile(synergy, np.linspace(0, 1, 1001))
        weights = self.pair_weights
        ordered = weights.reindex(weights.abs().sort_values(ascending=False).index)
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
        trained = list(getattr(self.report, "phi_columns", ()) or ())
        if trained and trained != list(PHI_COLUMNS):
            raise RuntimeError(
                f"the model was fitted on {len(trained)} pair columns and the code now"
                f" builds {len(PHI_COLUMNS)}, so it needs retraining"
            )
        return self.advantage_weights.reindex(PHI_COLUMNS).dropna()

    @property
    def informative(self) -> bool:
        if float(self.report.synergy_std) < MIN_SYNERGY_SPREAD:
            return False
        return self.report.advantage_gain > 0.0 and self.report.advantage_nulls_above == 0

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
