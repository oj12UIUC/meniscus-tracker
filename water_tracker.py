"""
Water-level / water-front tracking engine.

Reads a video file frame-by-frame (with optional sub-sampling), restricts
analysis to a user-defined ROI over the tube/pipette, and detects the
position of the water boundary (the meniscus) along the tube axis.

Scale comes from the tube's printed graduation marks when available (PRD §27):
the marks are detected once per video, their pixel spacing is measured, and the
stated value of one interval (mm or mL) turns meniscus movement into physical
units and cumulative absorbed volume. If marks can't be read, the engine falls
back to a manual `pixels_per_cm` (or raw pixels) and flags the run preliminary.

Outputs per video:
  - <stem>_data.csv         time, position, displacement, mm, marks, absorbed mL
  - <stem>_graph.png        time vs. meniscus movement (mm when calibrated, else px)
  - <stem>_absorption_ml.png  time vs. cumulative absorbed volume (when available)
  - <stem>_sorptivity.png   absorption vs. sqrt(time) with fitted slope (optional)
  - <stem>_meta.json        video + run metadata, scale source, confidence
  - verification/<stem>_frame######.jpg  overlay images (marks + meniscus + label)
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import graduation

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─── Detection primitives ────────────────────────────────────────────────────

def extract_strip(roi_gray: np.ndarray, orientation: str) -> np.ndarray:
    """
    Collapse a 2-D ROI into a 1-D intensity profile along the tube axis.

    Vertical setup  : average across columns -> profile indexed by row    (top->bottom)
    Horizontal setup: average across rows    -> profile indexed by column (left->right)
    """
    if orientation == "vertical":
        return roi_gray.mean(axis=1)
    elif orientation == "horizontal":
        return roi_gray.mean(axis=0)
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")


def detect_boundary(strip: np.ndarray, config: dict):
    """
    Find the water-boundary index inside a 1-D intensity strip.

    Returns int pixel index along the tube axis, or None if no boundary found.
    """
    feature_darker  = config.get("feature_darker", True)
    edge_type       = config.get("edge_type", "gradient")
    use_first_edge  = config.get("use_first_edge", True)
    search_start    = int(config.get("search_start", 0))
    search_end      = config.get("search_end", None)
    threshold_bias  = float(config.get("threshold_bias", 0))

    n = len(strip)
    if n == 0:
        return None

    s = max(0, min(search_start, n - 1))
    e = n if search_end is None else max(s + 1, min(int(search_end), n))
    region = strip[s:e].astype(float)
    if len(region) < 3:
        return None

    if edge_type == "gradient":
        # Light smoothing reduces noise from speckle / compression artifacts.
        kernel = np.ones(5) / 5.0
        smoothed_region = np.convolve(region, kernel, mode="same")
        grad = np.gradient(smoothed_region)
        # Mask the first/last few samples so np.gradient's boundary values
        # and any tube-tip / wall-edge artifacts can't dominate.
        pad = max(5, len(grad) // 50)
        if len(grad) > 2 * pad:
            grad_search = grad.copy()
            grad_search[:pad] = 0
            grad_search[-pad:] = 0
        else:
            grad_search = grad
        if feature_darker:
            idx = int(np.argmin(grad_search))   # bright -> dark transition
        else:
            idx = int(np.argmax(grad_search))   # dark   -> bright transition
        return s + idx

    if edge_type == "threshold":
        auto_thresh = float(np.mean(region)) + threshold_bias
        mask = region < auto_thresh if feature_darker else region > auto_thresh
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None
        return s + int(idxs[0] if use_first_edge else idxs[-1])

    raise ValueError(f"Unknown edge_type: {edge_type!r}")


def detect_diff_position(strip: np.ndarray, reference: np.ndarray, config: dict):
    """
    Frame-difference detector: locates the position along the strip where the
    intensity has changed most (vs. the very first sampled frame).

    Static features (printed markings, tape, tube edges, the wall behind the
    pipette) cancel out exactly because they don't change frame-to-frame. Only
    the moving water boundary survives. This is the right mode for videos with
    subtle/slow motion or strong fixed clutter inside the ROI.
    """
    search_start = int(config.get("search_start", 0))
    search_end = config.get("search_end", None)
    use_first_edge = config.get("use_first_edge", True)

    n = len(strip)
    if n == 0 or reference is None or len(reference) != n:
        return None

    s = max(0, min(search_start, n - 1))
    e = n if search_end is None else max(s + 1, min(int(search_end), n))

    diff = np.abs(strip[s:e].astype(float) - reference[s:e].astype(float))
    if len(diff) < 3:
        return None
    # Smooth the difference profile.
    kernel = np.ones(7) / 7.0
    diff = np.convolve(diff, kernel, mode="same")

    peak = float(np.max(diff))
    # Hold-previous behavior when the signal is too weak — this kills the
    # noisy oscillation in the first frames before the water has moved enough
    # to produce a clean diff signature.
    if peak < 8.0:
        return None    # caller will reuse last good position

    # Pixels considered "changed": those with diff above 30% of the peak.
    thresh = peak * 0.3
    idxs = np.where(diff > thresh)[0]
    if len(idxs) == 0:
        return s + int(np.argmax(diff))

    # The boundary is the first or last position in the changed region.
    return s + int(idxs[0] if use_first_edge else idxs[-1])


# ─── Scale / absorption helpers (PRD §27) ─────────────────────────────────────

def _absorption_sign(displacement: np.ndarray, direction: str) -> float:
    """
    Return +1/-1 so that absorbed movement reads as a positive quantity,
    independent of camera orientation (PRD §27.5).

    "increasing" : absorption moves the meniscus toward larger pixel index
    "decreasing" : absorption moves it toward smaller pixel index
    "auto"       : pick the sign from the net start->end movement
    """
    if direction == "increasing":
        return 1.0
    if direction == "decreasing":
        return -1.0
    # auto: positive if the meniscus ends at a larger index than it started.
    net = float(displacement[-1] - displacement[0]) if len(displacement) else 0.0
    return 1.0 if net >= 0 else -1.0


def _resolve_scale(config: dict, marks: dict) -> dict:
    """
    Decide how pixels map to physical units for this video.

    Priority (PRD §27 + manual failsafe):
      1. Manual one-unit calibration the user measured (manual_calibration) —
         honored first, because the user set it precisely when auto detection
         can't be trusted on faint/irregular marks.
      2. Detected graduation marks (when mark_calibration is on and the marks
         are regular enough).
      3. Manual pixels_per_cm.
      4. Raw pixels only.

    Returns a dict with:
        source       : "manual" | "marks" | "pixels"
        mm_per_px    : float | None
        ml_per_px    : float | None   (direct, when the span/marks are mL)
        spacing_px   : float | None   (px per stated interval/span)
        interval_unit: "mm" | "mL" | None
    """
    out = {"source": "pixels", "mm_per_px": None, "ml_per_px": None,
           "spacing_px": None, "interval_unit": None}

    # 1. Manual one-unit calibration (highest priority).
    span_px = config.get("manual_span_px")
    span_val = config.get("manual_span_value")
    if config.get("manual_calibration") and span_px and span_val and span_px > 0:
        unit = config.get("manual_span_unit", "mm")
        per_px = float(span_val) / float(span_px)
        out.update(source="manual", spacing_px=float(span_px), interval_unit=unit)
        if unit == "mL":
            out["ml_per_px"] = per_px
        else:
            out["mm_per_px"] = per_px
        return out

    # 2. Auto-detected graduation marks.
    if config.get("mark_calibration") and marks.get("ok"):
        spacing = marks["spacing_px"]
        unit = config.get("mark_interval_unit", "mm")
        scale = graduation.derive_scale(spacing, config.get("mark_interval_value", 1.0), unit)
        out.update(source="marks", spacing_px=spacing, interval_unit=unit)
        if unit == "mL":
            out["ml_per_px"] = scale["per_px"]
        else:  # length graduations
            out["mm_per_px"] = scale["per_px"]
        return out

    # 3. Manual pixels-per-cm fallback.
    ppc = config.get("pixels_per_cm")
    if ppc:
        out.update(source="pixels_per_cm", mm_per_px=10.0 / float(ppc), interval_unit="mm")
    return out


def _mm_to_ml_factor(config: dict) -> float | None:
    """mL per mm of meniscus travel, from the tube inner diameter (πr²·1mm)."""
    d = config.get("tube_diameter_mm")
    if not d or d <= 0:
        return None
    r_mm = float(d) / 2.0
    area_mm2 = np.pi * r_mm * r_mm
    return area_mm2 / 1000.0  # mm³ -> mL


# ─── Per-video analysis ──────────────────────────────────────────────────────

def analyze_video(video_path, config: dict, output_dir=None, verbose: bool = True,
                  progress_cb=None) -> dict:
    """
    Process a single video and (optionally) write outputs to disk.

    progress_cb : optional callable(fraction_0_to_1) invoked during the frame
                  loop — used by the web portal to drive a progress bar (§28.6).

    Returns a dict:
        df      : pandas DataFrame (time_s, position_px, ..., position_mm,
                  meniscus_position_marks, cumulative_absorbed_ml)
        meta    : dict of video + run metadata (incl. scale source + confidence)
        outputs : list of Path objects written (empty if output_dir is None)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    roi = config.get("roi")
    if roi is None:
        raise ValueError("ROI is required. Set 'roi': (x, y, w, h) in the config.")
    x, y, w, h = map(int, roi)

    orientation     = config.get("orientation", "vertical")
    sample_n        = max(1, int(config.get("sample_every_n_frames", 10)))
    smoothing_win   = int(config.get("smoothing_window", 0))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / fps if fps > 0 else 0.0

    if verbose:
        print(f"  Video:        {video_path.name}")
        print(f"  Resolution:   {frame_w}x{frame_h}  FPS: {fps:.2f}  "
              f"Frames: {total_frames}  Duration: {duration_s:.1f}s")
        print(f"  ROI:          x={x} y={y} w={w} h={h}  Orientation: {orientation}")
        print(f"  Sampling:     1 frame every {sample_n}  (~{sample_n/fps:.2f}s)")

    # Snapshot which sampled frames to keep as verification overlays.
    def _snap(f):
        return max(0, (f // sample_n) * sample_n)
    if total_frames > 0:
        verify_at = {_snap(0),
                     _snap(total_frames // 4),
                     _snap(total_frames // 2),
                     _snap(3 * total_frames // 4),
                     _snap(total_frames - 1)}
    else:
        verify_at = {0}

    times = []
    positions = []
    verification_raw = {}      # frame_idx -> (frame_copy, rx1, ry1, rx2, ry2, pos, t_s)
    mark_roi_samples = []      # gray ROI crops spanning the video, for mark detection
    reference_strip = None
    last_good_pos = None
    using_frame_diff = config.get("edge_type") == "frame_diff"

    frame_idx = 0
    next_progress = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if progress_cb is not None and total_frames > 0:
            frac = frame_idx / total_frames
            if frac >= next_progress:
                try:
                    progress_cb(min(1.0, frac))
                except Exception:
                    pass
                next_progress = frac + 0.02

        if frame_idx % sample_n == 0:
            ry1, ry2 = max(0, y), min(frame_h, y + h)
            rx1, rx2 = max(0, x), min(frame_w, x + w)

            if ry2 > ry1 and rx2 > rx1:
                roi_crop = frame[ry1:ry2, rx1:rx2]
                gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
                strip = extract_strip(gray, orientation)
                if using_frame_diff:
                    if reference_strip is None:
                        reference_strip = strip.copy()
                    pos = detect_diff_position(strip, reference_strip, config)
                    if pos is None and last_good_pos is not None:
                        pos = last_good_pos        # hold previous when signal too weak
                    elif pos is None:
                        pos = 0                    # initial frames: assume no motion yet
                    last_good_pos = pos
                else:
                    pos = detect_boundary(strip, config)
                t_s = frame_idx / fps

                # Collect ROI crops spanning the video for mark detection. The
                # meniscus sits at different positions in each, so a median over
                # them keeps the static marks sharp and averages the water out.
                if frame_idx in verify_at:
                    mark_roi_samples.append(gray.copy())

                if pos is not None:
                    times.append(t_s)
                    positions.append(pos)
                    if frame_idx in verify_at:
                        verification_raw[frame_idx] = (frame.copy(), rx1, ry1, rx2, ry2, pos, t_s)

        frame_idx += 1

    cap.release()
    if progress_cb is not None:
        try:
            progress_cb(1.0)
        except Exception:
            pass

    if not times:
        warnings.warn(f"No positions detected for {video_path.name}. "
                      f"Check ROI and detection parameters.")
        return {"df": pd.DataFrame(), "meta": {"video": video_path.name}, "outputs": []}

    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    displacement = positions - positions[0]

    smoothed = displacement.copy()
    if _HAS_SCIPY and smoothing_win >= 3 and len(displacement) >= smoothing_win:
        win = smoothing_win if smoothing_win % 2 == 1 else smoothing_win + 1
        try:
            smoothed = savgol_filter(displacement, window_length=win, polyorder=2)
        except Exception as exc:
            warnings.warn(f"Savitzky-Golay smoothing failed: {exc}")

    # ── Graduation-mark detection + scale resolution (PRD §27) ────────────────
    marks = {"ok": False, "positions_px": [], "spacing_px": 0.0,
             "n_marks": 0, "regularity": 0.0, "reason": "mark calibration off"}
    if config.get("mark_calibration") and mark_roi_samples:
        try:
            ref_roi = np.median(np.stack(mark_roi_samples, axis=0), axis=0).astype(np.uint8)
            marks = graduation.detect_marks(ref_roi, orientation, config)
        except Exception as exc:
            marks = {**marks, "reason": f"mark detection error: {exc}"}

    scale = _resolve_scale(config, marks)

    # ── Convert displacement into marks / mm / mL, oriented so absorption > 0 ──
    direction = config.get("absorption_direction", "auto")
    sign = _absorption_sign(smoothed, direction)
    absorbed_px = sign * smoothed          # positive-as-absorbed pixel travel

    n = len(displacement)
    nan = np.full(n, np.nan)

    if scale["spacing_px"]:
        meniscus_position_marks = absorbed_px / scale["spacing_px"]
    else:
        meniscus_position_marks = nan.copy()

    if scale["mm_per_px"] is not None:
        position_mm = absorbed_px * scale["mm_per_px"]
    else:
        position_mm = nan.copy()

    # Cumulative absorbed volume in mL.
    if scale["ml_per_px"] is not None:                 # volume graduations: direct
        cumulative_absorbed_ml = absorbed_px * scale["ml_per_px"]
    else:                                              # length graduations: via diameter
        mm_to_ml = _mm_to_ml_factor(config)
        if scale["mm_per_px"] is not None and mm_to_ml is not None:
            cumulative_absorbed_ml = position_mm * mm_to_ml
        else:
            cumulative_absorbed_ml = nan.copy()

    total_absorbed_ml = (float(np.nanmax(cumulative_absorbed_ml))
                         if np.isfinite(cumulative_absorbed_ml).any() else None)

    # ── Confidence tier (PRD §16 / §27.7) ─────────────────────────────────────
    movement_px = float(np.nanmax(np.abs(displacement))) if n else 0.0
    movement_ok = movement_px >= 3.0
    reliable_scale = (scale["source"] == "marks"
                      and marks.get("regularity", 0) >= graduation.MIN_REGULARITY
                      and marks.get("n_marks", 0) >= graduation.MIN_MARKS) \
        or scale["source"] in ("manual", "pixels_per_cm")
    confidence = "strong" if (reliable_scale and movement_ok) else "preliminary"
    conf_reasons = []
    if not movement_ok:
        conf_reasons.append("little/no meniscus movement detected")
    if scale["source"] == "marks" and not (marks.get("regularity", 0) >= graduation.MIN_REGULARITY):
        conf_reasons.append(marks.get("reason") or "irregular marks")
    if scale["source"] == "pixels":
        conf_reasons.append("no mark/manual scale — results in pixels only")

    # Legacy cm columns (kept for backward compatibility with v1 outputs).
    ppc = config.get("pixels_per_cm")
    if ppc:
        displacement_cm = displacement / float(ppc)
        smoothed_cm     = smoothed / float(ppc)
    else:
        displacement_cm = nan.copy()
        smoothed_cm     = nan.copy()

    df = pd.DataFrame({
        "time_s":                   times,
        "position_px":              positions,
        "displacement_px":          displacement,
        "smoothed_displacement_px": smoothed,
        "displacement_cm":          displacement_cm,
        "smoothed_displacement_cm": smoothed_cm,
        "meniscus_position_marks":  meniscus_position_marks,
        "position_mm":              position_mm,
        "cumulative_absorbed_ml":   cumulative_absorbed_ml,
    })

    meta = {
        "video":            video_path.name,
        "video_path":       str(video_path),
        "fps":              fps,
        "total_frames":     total_frames,
        "duration_s":       duration_s,
        "resolution":       f"{frame_w}x{frame_h}",
        "orientation":      orientation,
        "roi":              [x, y, w, h],
        "samples_taken":    len(times),
        "sample_every_n":   sample_n,
        "edge_type":        config.get("edge_type"),
        "feature_darker":   config.get("feature_darker"),
        "use_first_edge":   config.get("use_first_edge"),
        "smoothing_win":    smoothing_win,
        # Scale + marks (PRD §27)
        "scale_source":         scale["source"],
        "mark_spacing_px":      marks.get("spacing_px"),
        "n_marks":              marks.get("n_marks"),
        "mark_regularity":      marks.get("regularity"),
        "mark_interval_value":  config.get("mark_interval_value"),
        "mark_interval_unit":   config.get("mark_interval_unit"),
        "pixels_per_cm":        ppc,
        "mm_per_px":            scale["mm_per_px"],
        "ml_per_px":            scale["ml_per_px"],
        "absorption_direction": direction,
        "total_absorbed_ml":    total_absorbed_ml,
        # Confidence (PRD §16)
        "confidence":           confidence,
        "confidence_reasons":   conf_reasons,
        "marks_reason":         marks.get("reason"),
    }

    if verbose:
        print(f"  Scale:        {scale['source']}"
              + (f"  ({marks['n_marks']} marks @ {marks['spacing_px']:.1f}px, "
                 f"reg={marks['regularity']:.2f})" if scale['source'] == 'marks' else ""))
        print(f"  Confidence:   {confidence}"
              + (f"  ({'; '.join(conf_reasons)})" if conf_reasons else ""))
        if total_absorbed_ml is not None:
            print(f"  Absorbed:     {total_absorbed_ml:.4f} mL (total)")

    outputs = []
    if output_dir is not None:
        outputs = _write_outputs(output_dir, video_path, df, meta, config, marks,
                                 scale, times, position_mm, cumulative_absorbed_ml,
                                 displacement, smoothed, displacement_cm, smoothed_cm,
                                 smoothing_win, verification_raw, orientation,
                                 confidence, verbose)

    return {"df": df, "meta": meta, "outputs": outputs}


# ─── Output writers ──────────────────────────────────────────────────────────

def _write_outputs(output_dir, video_path, df, meta, config, marks, scale, times,
                   position_mm, cumulative_absorbed_ml, displacement, smoothed,
                   displacement_cm, smoothed_cm, smoothing_win, verification_raw,
                   orientation, confidence, verbose):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    outputs = []

    csv_path = output_dir / f"{stem}_data.csv"
    df.to_csv(csv_path, index=False)
    outputs.append(csv_path)

    # ── Primary movement graph: mm when calibrated, else cm, else px (§6) ─────
    fig, ax = plt.subplots(figsize=(10, 5))
    if np.isfinite(position_mm).any():
        ax.plot(times, position_mm, color="crimson", linewidth=2,
                label="Meniscus movement (mm)")
        ax.set_ylabel("Meniscus movement from start (mm)")
    elif np.isfinite(displacement_cm).any():
        ax.plot(times, displacement_cm, alpha=0.35, color="steelblue", label="Raw (cm)")
        ax.plot(times, smoothed_cm, color="crimson", linewidth=2,
                label=f"Smoothed (window={smoothing_win})")
        ax.set_ylabel("Displacement from start (cm)")
    else:
        ax.plot(times, displacement, alpha=0.35, color="steelblue", label="Raw (px)")
        ax.plot(times, smoothed, color="crimson", linewidth=2,
                label=f"Smoothed (window={smoothing_win})")
        ax.set_ylabel("Displacement from start (px)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Meniscus movement — {video_path.name}  [{confidence}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    graph_path = output_dir / f"{stem}_graph.png"
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    outputs.append(graph_path)

    # ── Cumulative absorption (mL) vs time (§27.6) ────────────────────────────
    if np.isfinite(cumulative_absorbed_ml).any():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, cumulative_absorbed_ml, color="seagreen", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative absorbed volume (mL)")
        ax.set_title(f"Absorption — {video_path.name}  [{confidence}]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        abs_path = output_dir / f"{stem}_absorption_ml.png"
        fig.savefig(abs_path, dpi=150)
        plt.close(fig)
        outputs.append(abs_path)

        # ── Optional sorptivity: absorption vs sqrt(time) with a fitted slope ─
        sorp = config.get("sorptivity") or {}
        if sorp.get("enabled"):
            sp = _sorptivity_plot(output_dir, stem, times, cumulative_absorbed_ml,
                                  sorp, video_path.name, confidence)
            if sp:
                outputs.append(sp)

    # ── Verification overlays: marks + meniscus + label (§14, §27.7) ──────────
    verify_dir = output_dir / "verification"
    verify_dir.mkdir(exist_ok=True)
    for fidx, (frame, rx1, ry1, rx2, ry2, pos, t_s) in sorted(verification_raw.items()):
        overlay = frame  # already a copy
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        # Detected graduation marks (faint gray lines) so the ruler is visible.
        for mp in marks.get("positions_px", []):
            if orientation == "vertical":
                my = ry1 + int(mp)
                if ry1 <= my <= ry2:
                    cv2.line(overlay, (rx1, my), (rx2, my), (180, 180, 180), 1)
            else:
                mx = rx1 + int(mp)
                if rx1 <= mx <= rx2:
                    cv2.line(overlay, (mx, ry1), (mx, ry2), (180, 180, 180), 1)
        # Tracked meniscus (red).
        if orientation == "vertical":
            ly = ry1 + int(pos)
            cv2.line(overlay, (rx1, ly), (rx2, ly), (0, 0, 255), 2)
        else:
            lx = rx1 + int(pos)
            cv2.line(overlay, (lx, ry1), (lx, ry2), (0, 0, 255), 2)
        label = (f"t={t_s:.1f}s  pos={int(pos)}px  scale={scale['source']}  "
                 f"[{confidence}]")
        cv2.putText(overlay, label, (rx1, max(25, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        vpath = verify_dir / f"{stem}_frame{fidx:06d}.jpg"
        cv2.imwrite(str(vpath), overlay)
        outputs.append(vpath)

    meta_path = output_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    outputs.append(meta_path)

    if verbose:
        print(f"  Outputs:      {output_dir}")
        print(f"  -> {csv_path.name},  {graph_path.name},  "
              f"{len(verification_raw)} verification frame(s)")

    return outputs


def _sorptivity_plot(output_dir, stem, times, absorbed_ml, sorp, video_name, confidence):
    """Absorption vs sqrt(time) with a linear fit over the configured window."""
    t_min = float(sorp.get("t_min_s", 0.0) or 0.0)
    t_max = sorp.get("t_max_s", None)
    sqrt_t = np.sqrt(times)
    mask = times >= t_min
    if t_max is not None:
        mask &= times <= float(t_max)
    mask &= np.isfinite(absorbed_ml)
    if mask.sum() < 2:
        return None

    slope, intercept = np.polyfit(sqrt_t[mask], absorbed_ml[mask], 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sqrt_t, absorbed_ml, "o", ms=3, alpha=0.4, color="seagreen", label="data")
    fit_x = np.array([sqrt_t[mask].min(), sqrt_t[mask].max()])
    ax.plot(fit_x, slope * fit_x + intercept, color="black", lw=2,
            label=f"fit: slope={slope:.4g} mL/√s")
    ax.set_xlabel("√time  (√s)")
    ax.set_ylabel("Cumulative absorbed volume (mL)")
    ax.set_title(f"Sorptivity — {video_name}  [{confidence}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{stem}_sorptivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
