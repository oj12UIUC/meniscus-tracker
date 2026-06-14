"""
Graduation-mark detection (PRD §27).

The tube/pipette carries printed tick marks at a regular interval. Instead of
assuming a pixels-per-cm number, we read the scale *from the marks themselves*:
detect the evenly spaced ticks inside the ROI, measure their average pixel
spacing, and combine that with the physical value of one interval (stated in the
config, e.g. "1 mm per mark" or "0.01 mL per mark") to get a per-video
conversion factor. This self-calibrates from the tube, so camera distance, zoom,
and minor perspective changes have less effect.

This module is pure (no file I/O) and self-contained so it can be reused by the
engine, the CLIs, and the web portal without circular imports.

Key function:
    detect_marks(roi_gray, orientation, config) -> dict
        {
          "ok":          bool,    # marks found and regular enough to trust
          "positions_px": [int],  # mark positions along the tube axis (in ROI coords)
          "spacing_px":   float,  # median spacing between consecutive marks
          "n_marks":      int,
          "regularity":   float,  # 0..1, 1 == perfectly even spacing
          "reason":       str,    # why ok is False, when applicable
        }
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a declared dependency
    _HAS_SCIPY = False


# Defaults for the regularity gate. A run with fewer marks than MIN_MARKS or a
# spacing coefficient-of-variation worse than (1 - MIN_REGULARITY) is treated as
# unreliable -> the caller falls back to manual calibration and flags the result
# "preliminary" (PRD §16, §27.7).
MIN_MARKS = 4
MIN_REGULARITY = 0.70


def profile_along_axis(roi_gray: np.ndarray, orientation: str) -> np.ndarray:
    """
    Collapse a 2-D ROI to a 1-D intensity profile along the tube axis.

    Mirrors water_tracker.extract_strip but kept local to avoid an import cycle.
    Vertical setup   -> average across columns -> indexed top->bottom.
    Horizontal setup -> average across rows    -> indexed left->right.
    """
    if orientation == "vertical":
        return roi_gray.mean(axis=1)
    if orientation == "horizontal":
        return roi_gray.mean(axis=0)
    raise ValueError(f"Unknown orientation: {orientation!r}")


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Edge-safe rolling mean used to remove slow lighting gradients."""
    window = max(3, int(window))
    if _HAS_SCIPY:
        return uniform_filter1d(x, size=window, mode="nearest")
    # Fallback: pad-and-convolve.
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="same")[pad:pad + len(x)]


def _dominant_period(signal: np.ndarray, min_lag: int, max_lag: int) -> float:
    """
    Estimate the dominant repeat period of a 1-D signal via autocorrelation.

    Returns the lag (in px) of the strongest autocorrelation peak inside
    [min_lag, max_lag], or 0.0 if none stands out. Used to set the minimum
    peak-to-peak distance so spurious closely spaced peaks (water boundary,
    speckle) don't get mistaken for marks.
    """
    n = len(signal)
    if n < 2 * min_lag:
        return 0.0
    s = signal - signal.mean()
    ac = np.correlate(s, s, mode="full")[n - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    hi = min(max_lag, n - 1)
    if hi <= min_lag:
        return 0.0
    best_lag, best_val = 0, 0.0
    for lag in range(min_lag, hi):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > best_val:
            best_lag, best_val = lag, ac[lag]
    return float(best_lag) if best_val > 0.1 else 0.0


def _refine_spacing(positions: np.ndarray) -> tuple[float, float]:
    """
    Given detected mark positions, return (spacing_px, regularity).

    Handles occasional missed marks: a gap that is ~2x the typical spacing is a
    single skipped tick, so we fold those back to the base spacing before scoring
    regularity rather than letting them wreck the coefficient of variation.
    """
    diffs = np.diff(positions).astype(float)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0, 0.0

    base = float(np.median(diffs))
    if base <= 0:
        return 0.0, 0.0

    # Fold near-integer multiples (1 or 2 skipped marks) down to the base.
    folded = []
    for d in diffs:
        k = round(d / base)
        if k >= 1 and abs(d - k * base) <= 0.35 * base:
            folded.append(d / k)
        else:
            folded.append(d)
    folded = np.asarray(folded, dtype=float)

    spacing = float(np.median(folded))
    cv = float(np.std(folded) / spacing) if spacing > 0 else 1.0
    regularity = max(0.0, 1.0 - cv)
    return spacing, regularity


def detect_marks(roi_gray: np.ndarray, orientation: str, config: dict) -> dict:
    """
    Detect evenly spaced graduation marks in an ROI (PRD §27.1).

    config keys used:
        marks_darker (bool)   : marks are darker than the tube background (default True)
        search_start (int)    : restrict detection along the axis
        search_end   (int|None)
        min_marks (int)       : override MIN_MARKS
        min_regularity (float): override MIN_REGULARITY
    """
    fail = {"ok": False, "positions_px": [], "spacing_px": 0.0,
            "n_marks": 0, "regularity": 0.0, "reason": ""}

    if not _HAS_SCIPY:
        return {**fail, "reason": "scipy not available for peak detection"}

    profile = profile_along_axis(roi_gray, orientation).astype(float)
    n = len(profile)
    if n < 3 * MIN_MARKS:
        return {**fail, "reason": "ROI too small along the tube axis"}

    s = int(config.get("search_start", 0) or 0)
    e = config.get("search_end", None)
    s = max(0, min(s, n - 1))
    e = n if e is None else max(s + 1, min(int(e), n))
    region = profile[s:e]
    if len(region) < 3 * MIN_MARKS:
        return {**fail, "reason": "search window too small"}

    marks_darker = bool(config.get("marks_darker", config.get("feature_darker", True)))
    min_marks = int(config.get("min_marks", MIN_MARKS))
    min_reg = float(config.get("min_regularity", MIN_REGULARITY))

    # Remove slow lighting gradient: marks are the high-frequency component.
    window = max(15, len(region) // 8)
    detrended = region - _rolling_mean(region, window)

    # Make marks appear as positive peaks regardless of polarity.
    signal = -detrended if marks_darker else detrended
    std = float(np.std(signal))
    if std < 1e-6:
        return {**fail, "reason": "flat profile (no contrast for marks)"}

    # Use autocorrelation to find the dominant repeat period, then require peaks
    # to be at least ~0.6x that period apart. This suppresses spurious closely
    # spaced peaks (water boundary, speckle) that otherwise wreck regularity.
    period = _dominant_period(signal, min_lag=5, max_lag=max(6, len(region) // min_marks))
    min_distance = max(3, int(0.6 * period)) if period > 0 else 3

    # Prominence scaled to the signal so it adapts to contrast.
    prominence = max(1.0, 0.4 * std)
    peaks, _ = find_peaks(signal, prominence=prominence, distance=min_distance)

    if len(peaks) < min_marks:
        return {**fail, "n_marks": int(len(peaks)),
                "reason": f"only {len(peaks)} mark candidate(s) found"}

    positions = peaks + s  # back to ROI-axis coordinates
    spacing, regularity = _refine_spacing(positions)

    ok = (len(positions) >= min_marks
          and regularity >= min_reg
          and spacing > 2.0)
    reason = "" if ok else (
        f"irregular marks (regularity={regularity:.2f}, n={len(positions)}, "
        f"spacing={spacing:.1f}px)")

    return {
        "ok": bool(ok),
        "positions_px": [int(p) for p in positions],
        "spacing_px": round(spacing, 3),
        "n_marks": int(len(positions)),
        "regularity": round(regularity, 3),
        "reason": reason,
    }


def derive_scale(spacing_px: float, interval_value: float, interval_unit: str) -> dict:
    """
    Convert detected mark spacing into a usable scale factor (PRD §27.2).

    Returns {"unit": "mm"|"mL", "per_px": value-per-pixel}. One pixel of meniscus
    travel equals `per_px` of the stated unit.
    """
    if spacing_px <= 0 or interval_value is None or interval_value <= 0:
        return {"unit": interval_unit, "per_px": None}
    return {"unit": interval_unit, "per_px": float(interval_value) / float(spacing_px)}


if __name__ == "__main__":
    # Quick self-test / probe: detect marks on a single ROI frame of a video.
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="Probe graduation-mark detection on one frame")
    ap.add_argument("video")
    ap.add_argument("--roi", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--orientation", choices=["vertical", "horizontal"], default="horizontal")
    ap.add_argument("--frame", type=int, default=None, help="frame index (default: middle)")
    ap.add_argument("--marks-darker", choices=["true", "false"], default="true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame if args.frame is not None else total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("could not read frame")

    x, y, w, h = args.roi
    roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    cfg = {"marks_darker": args.marks_darker == "true"}
    res = detect_marks(roi, args.orientation, cfg)
    print(res)
