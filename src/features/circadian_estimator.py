"""
src/features/circadian_estimator.py
────────────────────────────────────────────────────────────────────────────────
Circadian Phase Estimator — Core Novel Component
circadian-mci-predictor | PhD Research

Fits a cosine model to passive screen-on event timestamps and derives:
  - Circadian phase (estimated peak activity hour)
  - Amplitude (rhythm strength)
  - Goodness-of-fit (R²)
  - Circadian Regularity Score (CRS)
  - Phase drift rate (slope of phase change over time)

Mathematical model:
    f(t) = A · cos( 2π(t − φ) / 24 ) + C

where:
    φ = circadian phase (hours, 0–24)
    A = amplitude
    C = baseline offset

Usage:
    from src.features.circadian_estimator import CircadianEstimator
    ce = CircadianEstimator(window=14)
    circ_df = ce.fit_transform(feature_df)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

__all__ = ["CircadianEstimator"]

# ── Constants ─────────────────────────────────────────────────────────────────
HEALTHY_PHASE_STD_REF = 0.8   # hours — reference SD for a regular sleeper
MAX_PHASE_STD         = 3.0   # hours — upper bound for CRS normalisation
DEFAULT_WINDOW        = 14    # days for phase estimation window
MIN_EVENTS_FOR_FIT    = 5     # minimum screen events to attempt curve fit


# ── Cosine model ──────────────────────────────────────────────────────────────
def _cosine_model(t: np.ndarray, amplitude: float, phase: float, offset: float) -> np.ndarray:
    """24-hour cosine circadian rhythm model."""
    return amplitude * np.cos(2 * np.pi * (t - phase) / 24) + offset


# ── Core estimator ────────────────────────────────────────────────────────────
class CircadianEstimator:
    """
    Estimates circadian rhythm features from daily screen-on peak hours.

    Parameters
    ----------
    window : int
        Sliding window size in days for phase estimation (default: 14).
    std_ref : float
        Healthy-baseline phase standard deviation in hours used to
        normalise the Circadian Regularity Score (default: 0.8 hrs).
    drift_window : int
        Window size in days for computing the linear phase-drift rate
        (default: 30).
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        std_ref: float = HEALTHY_PHASE_STD_REF,
        drift_window: int = 30,
    ) -> None:
        self.window       = window
        self.std_ref      = std_ref
        self.drift_window = drift_window

    # ── Public API ────────────────────────────────────────────────────────────
    def fit_transform(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-day circadian features for all participants.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Must contain columns:
              - participant_id  (int or str)
              - day             (int, 0-based)
              - screen_peak_hour (float, 0–23.99)

        Returns
        -------
        pd.DataFrame with columns:
            participant_id, day,
            circadian_phase_est, circadian_amplitude, circadian_r2,
            circadian_phase_std_14d, circadian_phase_std_30d,
            circadian_regularity_score, phase_drift_rate
        """
        self._validate(feature_df)
        parts = []
        for pid, grp in tqdm(
            feature_df.groupby("participant_id"),
            desc="Estimating circadian phases",
            unit="participant",
        ):
            parts.append(self._process_participant(pid, grp))
        return pd.concat(parts, ignore_index=True)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _validate(self, df: pd.DataFrame) -> None:
        required = {"participant_id", "day", "screen_peak_hour"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"feature_df is missing columns: {missing}")

    def _process_participant(self, pid: int | str, grp: pd.DataFrame) -> pd.DataFrame:
        grp   = grp.sort_values("day").copy()
        days  = grp["day"].values
        hours = grp["screen_peak_hour"].values

        phases, amplitudes, r2s = self._estimate_phases(days, hours, grp)

        phase_series = pd.Series(phases, dtype=float)

        phase_std_14 = phase_series.rolling(14, min_periods=3).std().fillna(0)
        phase_std_30 = phase_series.rolling(30, min_periods=7).std().fillna(0)

        crs        = self._compute_crs(phase_std_14)
        drift_rate = self._compute_drift_rate(phase_series)

        return pd.DataFrame({
            "participant_id":             pid,
            "day":                        days,
            "circadian_phase_est":        np.round(phases,      4),
            "circadian_amplitude":        np.round(amplitudes,  4),
            "circadian_r2":               np.round(r2s,         4),
            "circadian_phase_std_14d":    np.round(phase_std_14.values, 4),
            "circadian_phase_std_30d":    np.round(phase_std_30.values, 4),
            "circadian_regularity_score": np.round(crs.values,  4),
            "phase_drift_rate":           np.round(drift_rate.values, 6),
        })

    def _estimate_phases(
        self,
        days: np.ndarray,
        hours: np.ndarray,
        grp: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phases, amplitudes, r2s = [], [], []
        for day in days:
            mask = (grp["day"] >= day - self.window) & (grp["day"] <= day)
            window_hours = grp.loc[mask, "screen_peak_hour"].dropna().values
            result = self._fit_cosine(window_hours)
            phases.append(result["phase"])
            amplitudes.append(result["amplitude"])
            r2s.append(result["r2"])
        return np.array(phases), np.array(amplitudes), np.array(r2s)

    @staticmethod
    def _fit_cosine(hours: np.ndarray) -> dict:
        """
        Fit cosine model to hourly histogram of screen-on events.

        Returns dict with keys: phase, amplitude, r2.
        Returns NaN values if fit fails or insufficient data.
        """
        nan_result = {"phase": np.nan, "amplitude": np.nan, "r2": np.nan}
        if len(hours) < MIN_EVENTS_FOR_FIT:
            return nan_result

        counts, edges = np.histogram(hours, bins=24, range=(0, 24))
        bin_centers   = (edges[:-1] + edges[1:]) / 2
        counts        = counts.astype(float)

        peak_bin = bin_centers[counts.argmax()]
        p0 = [counts.max() / 2, peak_bin, counts.mean()]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _cosine_model,
                    bin_centers,
                    counts,
                    p0=p0,
                    bounds=(
                        [-np.inf, 0,  0],
                        [ np.inf, 24, np.inf],
                    ),
                    maxfev=2000,
                )
            amplitude, phase, _ = popt

            fitted = _cosine_model(bin_centers, *popt)
            ss_res = float(np.sum((counts - fitted) ** 2))
            ss_tot = float(np.sum((counts - counts.mean()) ** 2))
            r2     = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

            return {
                "phase":     float(np.clip(phase, 0, 24)),
                "amplitude": float(amplitude),
                "r2":        float(np.clip(r2, -1, 1)),
            }
        except (RuntimeError, ValueError):
            return nan_result

    def _compute_crs(self, phase_std_series: pd.Series) -> pd.Series:
        """
        Circadian Regularity Score:
            CRS = 1 − phase_std / (MAX_PHASE_STD)
        Clipped to [0, 1]; higher = more regular.
        """
        return (1 - phase_std_series / MAX_PHASE_STD).clip(0, 1)

    def _compute_drift_rate(self, phase_series: pd.Series) -> pd.Series:
        """
        Linear slope of circadian phase over a rolling drift_window.
        Positive slope = phase advancing (earlier peak each day).
        Negative slope = phase delaying (later peak each day).
        """
        def _slope(arr: pd.Series) -> float:
            arr = arr.dropna()
            if len(arr) < 5:
                return 0.0
            x = np.arange(len(arr), dtype=float)
            return float(np.polyfit(x, arr.values, 1)[0])

        return phase_series.rolling(self.drift_window, min_periods=7).apply(
            _slope, raw=False
        ).fillna(0.0)