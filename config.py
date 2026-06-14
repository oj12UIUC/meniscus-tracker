"""
Per-setup configuration profiles for the water-level tracker.

Each profile defines the ROI, orientation, detection parameters, and (PRD §27)
graduation-mark calibration for a specific experimental layout. ROI values are
PLACEHOLDERS unless noted — run roi_picker.py on a representative video to
determine correct values for your framing.

ROI format: (x, y, width, height) in pixels.

Graduation-mark calibration (PRD §27)
-------------------------------------
The tube carries printed tick marks at a regular interval. When
`mark_calibration` is True, the engine detects those marks inside the ROI,
measures their pixel spacing, and combines it with the stated physical value of
one interval to convert meniscus movement into mm and/or mL — no separately
measured pixels-per-cm needed. Relevant keys:

    mark_calibration   : turn mark-based scaling on for this setup
    mark_interval_value: physical value of ONE mark interval (e.g. 1.0)
    mark_interval_unit : "mm" (length graduations) or "mL" (volume graduations)
    marks_darker       : True if the printed ticks are darker than the tube
    tube_diameter_mm   : inner diameter; only needed to turn mm marks into mL
                         (when the unit is "mL", volume is reported directly)
    absorption_direction: "auto" | "increasing" | "decreasing" — which way the
                         meniscus pixel index moves as water is absorbed, so the
                         reported absorbed volume is always positive (§27.5).
                         "auto" picks the sign from the net movement.
    sorptivity         : optional ASTM-C1585-style absorption-vs-sqrt(time) fit
                         {"enabled": bool, "t_min_s": float, "t_max_s": float|None}

If marks are missing, too few, or irregular, the engine flags the run
"preliminary" and falls back to `pixels_per_cm` (or raw pixels). These interval
values are STARTING POINTS — confirm them against a verification overlay for
each setup before trusting the mm/mL numbers.
"""

# Defaults shared by every profile; individual profiles override as needed.
_MARK_DEFAULTS = {
    "mark_calibration":     False,
    "mark_interval_value":  1.0,
    "mark_interval_unit":   "mm",      # "mm" or "mL"
    "marks_darker":         True,
    "tube_diameter_mm":     None,      # set to convert mm -> mL
    "absorption_direction": "auto",    # "auto" | "increasing" | "decreasing"
    "sorptivity":           {"enabled": False, "t_min_s": 0.0, "t_max_s": None},
    # Manual calibration failsafe (when auto mark detection is unreliable).
    # Measure a known span on a frame with calibrate.py or the portal: how many
    # pixels (`manual_span_px`) equal a stated value (`manual_span_value`) in
    # `manual_span_unit`. When set, this takes precedence over mark detection.
    "manual_calibration":   False,
    "manual_span_px":       None,      # pixels measured for the span
    "manual_span_value":    None,      # what that span represents (e.g. 5)
    "manual_span_unit":     "mm",      # "mm" or "mL"
}


def _profile(**overrides) -> dict:
    """Build a profile from the mark-calibration defaults plus per-setup keys."""
    p = dict(_MARK_DEFAULTS)
    p.update(overrides)
    return p


SETUP_PROFILES = {

    # Burgess vertical tube — water level changes, scanned top->bottom
    "burgess_vertical": _profile(
        description="Burgess vertical tube — scanned top->bottom",
        orientation="vertical",
        roi=(704, 936, 51, 664),        # Burg_T1.MOV calibration (tightened)
        feature_darker=True,            # water is darker than air above
        edge_type="frame_diff",         # robust to markings/tape; alt: 'gradient', 'threshold'
        use_first_edge=True,            # topmost edge of the changed region
        search_start=0,
        search_end=None,
        threshold_bias=0,
        smoothing_window=15,            # Savitzky-Golay window, must be odd, 0 = off
        sample_every_n_frames=15,       # 15 @ 30fps ≈ 0.5s spacing
        pixels_per_cm=None,             # fallback scale if marks can't be read
        # Graduation marks (CONFIRM interval against an overlay before trusting):
        mark_calibration=True,
        mark_interval_value=1.0,
        mark_interval_unit="mm",
        marks_darker=True,
    ),

    # C202 horizontal pipette — water front advances left->right
    "c202_horizontal": _profile(
        description="C202 horizontal pipette — water front advances left->right",
        orientation="horizontal",
        roi=(80, 180, 750, 120),        # PLACEHOLDER
        feature_darker=True,
        edge_type="gradient",
        use_first_edge=False,           # rightmost edge = advancing front
        search_start=0,
        search_end=None,
        threshold_bias=0,
        smoothing_window=15,
        sample_every_n_frames=15,
        pixels_per_cm=None,
        mark_calibration=True,
        mark_interval_value=1.0,
        mark_interval_unit="mm",
        marks_darker=True,
    ),

    # NP vertical pipette, filmed in portrait (tube runs top->bottom in frame).
    # Newer reliable NP videos (BER_30_*, IMG_*) are 1080x1920 portrait.
    "np_vertical": _profile(
        description="NP vertical pipette (portrait video) — scanned top->bottom",
        orientation="vertical",
        roi=(611, 40, 80, 1800),        # default for 1080x1920 (BER_30_T1) — tune per video
        feature_darker=True,
        edge_type="frame_diff",         # robust to faint printed markings
        use_first_edge=True,            # topmost edge of the changed region
        search_start=0,
        search_end=None,
        threshold_bias=0,
        smoothing_window=15,
        sample_every_n_frames=15,
        pixels_per_cm=None,
        mark_calibration=True,
        mark_interval_value=1.0,
        mark_interval_unit="mm",
        marks_darker=True,
    ),

    # Fly_Ash horizontal pipette
    "fly_ash_horizontal": _profile(
        description="Fly_Ash horizontal pipette — water front advances left->right",
        orientation="horizontal",
        roi=(339, 512, 1347, 97),       # 22_ISR4_623_T2.mp4 calibration (tighter v3)
        feature_darker=True,
        edge_type="frame_diff",         # robust to printed markings; alt: 'gradient', 'threshold'
        use_first_edge=False,
        search_start=0,
        search_end=None,
        threshold_bias=0,
        smoothing_window=15,
        sample_every_n_frames=15,
        pixels_per_cm=1100,             # fallback: 22_ISR4_623_T2 (1325 px ≈ 12 mm manual ground truth)
        # Marks validated on 22_ISR4_623_T2: ~103 px spacing ≈ 1 mm (matches px/cm above).
        mark_calibration=True,
        mark_interval_value=1.0,
        mark_interval_unit="mm",
        marks_darker=True,
    ),

    # CEMEX vertical tube (369-* videos) — NEW, ROI + marks need calibration
    "cemex_vertical": _profile(
        description="CEMEX vertical tube (369-* series) — scanned top->bottom",
        orientation="vertical",
        roi=None,                       # REQUIRED — run roi_picker.py on a 369-* video
        feature_darker=True,
        edge_type="frame_diff",
        use_first_edge=True,
        search_start=0,
        search_end=None,
        threshold_bias=0,
        smoothing_window=15,
        sample_every_n_frames=15,
        pixels_per_cm=None,
        mark_calibration=True,
        mark_interval_value=1.0,
        mark_interval_unit="mm",
        marks_darker=True,
    ),

    # Holcim vertical tube (PLC_Holcim_* videos) — NEW, ROI + marks need calibration
    "holcim_vertical": _profile(
        description="Holcim vertical tube (PLC_Holcim_* series) — scanned top->bottom",
        orientation="vertical",
        roi=None,                       # REQUIRED — run roi_picker.py on a PLC_Holcim_* video
        feature_darker=True,
        edge_type="frame_diff",
        use_first_edge=True,
        search_start=0,
        search_end=None,
        threshold_bias=0,
        smoothing_window=15,
        sample_every_n_frames=15,
        pixels_per_cm=None,
        mark_calibration=True,
        mark_interval_value=1.0,
        mark_interval_unit="mm",
        marks_darker=True,
    ),

    # General-purpose profile — configure before use
    "general": _profile(
        description="General-purpose profile — set roi before running",
        orientation="vertical",
        roi=None,                       # REQUIRED — must be set by user
        feature_darker=True,
        edge_type="gradient",
        use_first_edge=True,
        search_start=0,
        search_end=None,
        threshold_bias=0,
        smoothing_window=11,
        sample_every_n_frames=10,
        pixels_per_cm=None,
        # Mark calibration off by default; enable + set interval for your tube.
        mark_calibration=False,
        mark_interval_value=1.0,
        mark_interval_unit="mm",
        marks_darker=True,
    ),
}

# Auto-profile mapping: folder name (case sensitive) -> profile key
FOLDER_PROFILE_MAP = {
    "Burgess": "burgess_vertical",
    "C202":    "c202_horizontal",
    "NP":      "np_vertical",
    "Fly_Ash": "fly_ash_horizontal",
    "CEMEX":   "cemex_vertical",
    "Holcim":  "holcim_vertical",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
