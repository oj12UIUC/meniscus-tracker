"""
Single-page web portal for the water-level / meniscus tracker (PRD §28).

A researcher uploads one or more videos, picks a setup profile (and optionally
edits the ROI / detection / mark-calibration settings), runs the analysis, and
sees the graphs, time-series table, verification overlays, and confidence label
rendered on the same page — with download buttons for the CSV, graphs, overlays,
and a full zip.

The portal contains NO detection or calculation logic of its own: it calls the
existing engine `water_tracker.analyze_video` and the shared `config` profiles
(PRD §28.4 — single source of logic).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import SETUP_PROFILES, VIDEO_EXTENSIONS
from water_tracker import analyze_video

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as st_img_coords
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

from PIL import Image, ImageDraw

st.set_page_config(page_title="Meniscus Tracker", layout="wide")

ACCEPTED = sorted(ext.lstrip(".") for ext in VIDEO_EXTENSIONS)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def zip_dir(folder: Path) -> bytes:
    """Zip an output folder into an in-memory buffer for a single download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in folder.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(folder))
    buf.seek(0)
    return buf.read()


def confidence_badge(meta: dict):
    conf = meta.get("confidence", "unknown")
    reasons = meta.get("confidence_reasons") or []
    if conf == "strong":
        st.success(f"Confidence: **strong**  ·  scale source: {meta.get('scale_source')}")
    else:
        st.warning(f"Confidence: **preliminary**  ·  scale source: {meta.get('scale_source')}"
                   + (f"  ·  {'; '.join(reasons)}" if reasons else ""))


def find_output(outputs, suffix: str):
    for p in outputs:
        if p.name.endswith(suffix):
            return p
    return None


@st.cache_data(show_spinner=False)
def grab_frame(file_bytes: bytes, name: str, frac: float):
    """Decode one RGB frame at `frac` of the video for previews/calibration."""
    tmp = Path(tempfile.gettempdir()) / f"_preview_{abs(hash(name)) % 10**8}_{name}"
    tmp.write_bytes(file_bytes)
    cap = cv2.VideoCapture(str(tmp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, min(frac, 0.999)) * total))
    ok, frame = cap.read()
    cap.release()
    try:
        tmp.unlink()
    except OSError:
        pass
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ─── Sidebar: setup & detection configuration (§28.3) ─────────────────────────

st.title("Meniscus Tracking & Absorption Portal")
st.caption("Upload experiment videos → track the meniscus against the tube's "
           "graduation marks → get movement (mm), absorbed volume (mL), graphs, "
           "and verification overlays. Wraps the same engine used by the CLIs.")

# Sidebar widget keys that hold per-profile settings. When the profile changes
# we clear them so the widgets re-initialize from the newly selected profile
# (otherwise Streamlit keeps the previous profile's ROI — the bug that left the
# box on the wall after switching setups).
_PROFILE_KEYS = ["w_orient", "w_rx", "w_ry", "w_rw", "w_rh", "w_edge", "w_fdark",
                 "w_first", "w_samp", "w_smooth", "w_markcal", "w_markval",
                 "w_markunit", "w_marksdark", "w_tubed", "w_absdir", "w_ppc"]

with st.sidebar:
    st.header("Setup")
    profile_name = st.selectbox("Setup profile", list(SETUP_PROFILES.keys()),
                                help="Pick a saved setup; its ROI + settings load below.")
    if st.session_state.get("_last_profile") != profile_name:
        for k in _PROFILE_KEYS + ["cal_pts", "roi_pts", "ui_roi", "ui_span_px",
                                  "ui_men_start", "ui_click_last"]:
            st.session_state.pop(k, None)         # reload fields for the new profile
        st.session_state["_last_profile"] = profile_name
    base = SETUP_PROFILES[profile_name].copy()

    st.caption(base.get("description", ""))

    roi = base.get("roi") or (0, 0, 0, 0)
    st.radio("Orientation", ["vertical", "horizontal"],
             index=0 if base.get("orientation") == "vertical" else 1,
             horizontal=True, key="w_orient")

    st.subheader("Region of interest (px)")
    c1, c2 = st.columns(2)
    c1.number_input("x", value=int(roi[0]), step=1, key="w_rx")
    c2.number_input("y", value=int(roi[1]), step=1, key="w_ry")
    c1.number_input("width", value=int(roi[2]), step=1, min_value=0, key="w_rw")
    c2.number_input("height", value=int(roi[3]), step=1, min_value=0, key="w_rh")

    st.subheader("Detection")
    st.selectbox("Edge type", ["frame_diff", "gradient", "threshold"],
                 index=["frame_diff", "gradient", "threshold"].index(
                     base.get("edge_type", "frame_diff")), key="w_edge")
    st.checkbox("Feature darker than background",
                value=bool(base.get("feature_darker", True)), key="w_fdark")
    st.checkbox("Use first (vs last) qualifying edge",
                value=bool(base.get("use_first_edge", True)), key="w_first")
    st.number_input("Sample every N frames", value=int(base.get("sample_every_n_frames", 15)),
                    min_value=1, step=1, key="w_samp",
                    help="Higher = faster / coarser. Time stays correct.")
    st.number_input("Smoothing window (odd, 0=off)",
                    value=int(base.get("smoothing_window", 15)), min_value=0, step=2, key="w_smooth")

    st.subheader("Graduation-mark calibration (§27)")
    st.checkbox("Use graduation marks for scale",
                value=bool(base.get("mark_calibration", False)), key="w_markcal")
    st.number_input("Value of one mark interval",
                    value=float(base.get("mark_interval_value", 1.0)),
                    min_value=0.0, step=0.01, format="%.4f", key="w_markval")
    st.selectbox("Interval unit", ["mm", "mL"],
                 index=0 if base.get("mark_interval_unit", "mm") == "mm" else 1, key="w_markunit")
    st.checkbox("Marks darker than tube", value=bool(base.get("marks_darker", True)), key="w_marksdark")
    st.number_input("Tube inner diameter (mm, for mm→mL; 0=unset)",
                    value=float(base.get("tube_diameter_mm") or 0.0),
                    min_value=0.0, step=0.1, key="w_tubed")
    st.selectbox("Absorption direction", ["auto", "increasing", "decreasing"],
                 index=["auto", "increasing", "decreasing"].index(
                     base.get("absorption_direction", "auto")), key="w_absdir")
    st.number_input("Fallback pixels-per-cm (0=unset)",
                    value=float(base.get("pixels_per_cm") or 0.0), min_value=0.0, step=1.0, key="w_ppc")

def build_config() -> dict:
    cfg = SETUP_PROFILES[profile_name].copy()
    ss = st.session_state
    orientation = ss["w_orient"]
    # ROI: the box drawn on the frame wins over the sidebar numbers.
    if ss.get("ui_roi"):
        roi = tuple(int(v) for v in ss["ui_roi"])
    else:
        roi = (int(ss["w_rx"]), int(ss["w_ry"]), int(ss["w_rw"]), int(ss["w_rh"]))
    cfg.update({
        "orientation": orientation,
        "roi": roi,
        "edge_type": ss["w_edge"],
        "feature_darker": ss["w_fdark"],
        "use_first_edge": ss["w_first"],
        "sample_every_n_frames": int(ss["w_samp"]),
        "smoothing_window": int(ss["w_smooth"]),
        "mark_calibration": ss["w_markcal"],
        "mark_interval_value": float(ss["w_markval"]),
        "mark_interval_unit": ss["w_markunit"],
        "marks_darker": ss["w_marksdark"],
        "tube_diameter_mm": (float(ss["w_tubed"]) if ss["w_tubed"] > 0 else None),
        "absorption_direction": ss["w_absdir"],
        "pixels_per_cm": (float(ss["w_ppc"]) if ss["w_ppc"] > 0 else None),
    })
    # Meniscus-start line: restrict detection to from there onward along the axis
    # (relative to the ROI), so the tracker ignores labels/clutter above the start.
    men = ss.get("ui_men_start")
    if men is not None:
        ref = roi[1] if orientation == "vertical" else roi[0]
        cfg["search_start"] = max(0, int(men) - int(ref))
    # Manual one-unit calibration from the two dots — overrides auto marks.
    if ss.get("ui_span_px", 0) > 0 and ss.get("man_value", 0) > 0:
        cfg.update({
            "manual_calibration": True,
            "manual_span_px": float(ss["ui_span_px"]),
            "manual_span_value": float(ss["man_value"]),
            "manual_span_unit": ss.get("man_unit", "mm"),
        })
    return cfg


# ─── Main: upload & run (§28.1, §28.6) ────────────────────────────────────────

uploads = st.file_uploader(
    "Upload video(s)", type=ACCEPTED, accept_multiple_files=True,
    help=f"Accepted: {', '.join('.' + e for e in ACCEPTED)}. Drag and drop works too.")


# ─── Interactive setup: draw ROI, calibrate, mark meniscus start (on the frame) ─

MODE_ROI = "① Draw ROI box (click 2 opposite corners)"
MODE_CAL = "② Calibrate scale (click 2 dots on a marking)"
MODE_MEN = "③ Mark meniscus START (click 1 point)"


def interactive_setup_ui(uploads):
    """Set the ROI box, the pixel→unit calibration, and the meniscus start by
    clicking directly on a preview frame."""
    ss = st.session_state
    with st.expander("🎯 Interactive setup — draw on the frame "
                     "(ROI box · calibration dots · meniscus start)", expanded=True):
        if not _HAS_CLICK:
            st.info("Interactive drawing needs `streamlit-image-coordinates` "
                    "(in requirements.txt). Use the sidebar ROI numbers instead.")
            return

        names = [u.name for u in uploads]
        sel = st.selectbox("Frame from", names, key="ui_file")
        up = next(u for u in uploads if u.name == sel)
        frac = st.slider("Frame position (scrub to where the water is moving)",
                         0.0, 1.0, 0.5, 0.01, key="ui_frac")
        frame = grab_frame(up.getbuffer().tobytes(), up.name, frac)
        if frame is None:
            st.warning("Could not read a frame from this file.")
            return
        H, W = frame.shape[:2]
        orientation = ss.get("w_orient", "vertical")

        mode = st.radio("What do you want to place?", [MODE_ROI, MODE_CAL, MODE_MEN], key="ui_mode")
        zdef = round(min(1.0, 900.0 / max(H, W)), 1)
        zoom = st.slider("Zoom (enlarge so faint marks are easy to click)",
                         0.3, 4.0, ss.get("ui_zoom", zdef), 0.1, key="ui_zoom")

        b = st.columns(4)
        if b[0].button("↺ ROI"):
            ss.pop("ui_roi", None); ss["roi_pts"] = []
        if b[1].button("↺ Dots"):
            ss.pop("ui_span_px", None); ss["cal_pts"] = []
        if b[2].button("↺ Meniscus"):
            ss.pop("ui_men_start", None)
        if b[3].button("↺ All"):
            for k in ("ui_roi", "ui_span_px", "ui_men_start", "roi_pts", "cal_pts"):
                ss.pop(k, None)

        # ── Compose the annotated, zoomed frame ───────────────────────────────
        dw, dh = int(W * zoom), int(H * zoom)
        img = Image.fromarray(frame).resize((dw, dh))
        draw = ImageDraw.Draw(img)
        lw = max(2, int(min(dw, dh) // 250))

        roi = ss.get("ui_roi")
        if roi:                                            # green ROI box
            rx0, ry0, rw0, rh0 = roi
            draw.rectangle([rx0 * zoom, ry0 * zoom, (rx0 + rw0) * zoom, (ry0 + rh0) * zoom],
                           outline=(0, 255, 0), width=lw)
        for px, py in ss.get("roi_pts", []):               # ROI corner in progress
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(0, 255, 0), width=lw)
        cp = ss.get("cal_pts", [])                         # red calibration dots
        for px, py in cp:
            draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=(255, 0, 0))
        if len(cp) == 2:
            draw.line([cp[0], cp[1]], fill=(255, 0, 0), width=lw)
        men = ss.get("ui_men_start")                       # blue meniscus-start line
        if men is not None:
            if orientation == "vertical":
                draw.line([0, men * zoom, dw, men * zoom], fill=(0, 80, 255), width=lw + 1)
            else:
                draw.line([men * zoom, 0, men * zoom, dh], fill=(0, 80, 255), width=lw + 1)

        hint = {MODE_ROI: "Click one corner of the tube region, then the opposite corner.",
                MODE_CAL: "Click the start of a marking, then the end of a known span.",
                MODE_MEN: "Click where the water meniscus STARTS."}[mode]
        st.caption(hint)
        click = st_img_coords(img, key="ui_canvas")
        if click is not None:
            c = (int(click["x"]), int(click["y"]))
            if ss.get("ui_click_last") != c:               # a fresh click
                ss["ui_click_last"] = c
                ox, oy = c[0] / zoom, c[1] / zoom          # back to original px
                if mode == MODE_ROI:
                    pts = ss.get("roi_pts", [])
                    if len(pts) >= 2:
                        pts = []
                    pts.append(c)
                    ss["roi_pts"] = pts
                    if len(pts) == 2:
                        x1, y1 = pts[0][0] / zoom, pts[0][1] / zoom
                        x2, y2 = pts[1][0] / zoom, pts[1][1] / zoom
                        ss["ui_roi"] = (int(min(x1, x2)), int(min(y1, y2)),
                                        int(abs(x2 - x1)), int(abs(y2 - y1)))
                        ss["roi_pts"] = []
                elif mode == MODE_CAL:
                    pts = ss.get("cal_pts", [])
                    if len(pts) >= 2:
                        pts = []
                    pts.append(c)
                    ss["cal_pts"] = pts
                    if len(pts) == 2:
                        d = ((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2) ** 0.5
                        ss["ui_span_px"] = round(d / zoom, 1)
                else:  # MODE_MEN
                    ss["ui_men_start"] = int(oy if orientation == "vertical" else ox)
                st.rerun()

        # ── Calibration value/unit + live status ──────────────────────────────
        st.markdown("**Calibration: what does the span between your 2 red dots represent?**")
        v1, v2 = st.columns(2)
        v1.number_input("Span equals (value)", min_value=0.0, step=0.01, key="man_value")
        v2.selectbox("Unit", ["mm", "mL"], key="man_unit")

        st.markdown("**Current setup**")
        st.write(f"🟩 ROI box: `{roi}`" if roi else "🟩 ROI box: _not set — draw it, or use the sidebar numbers_")
        if ss.get("ui_span_px", 0) > 0 and ss.get("man_value", 0) > 0:
            per = ss["man_value"] / ss["ui_span_px"]
            st.write(f"🔴 Scale: {ss['ui_span_px']:.1f} px = {ss['man_value']} "
                     f"{ss['man_unit']}  →  **{per:.5g} {ss['man_unit']}/px** (manual, overrides auto)")
        elif ss.get("ui_span_px", 0) > 0:
            st.write(f"🔴 Span measured: {ss['ui_span_px']:.1f} px — now enter what it equals above.")
        else:
            st.write("🔴 Scale: _not set — auto mark detection will be used_")
        st.write(f"🔵 Meniscus start: axis pixel `{men}`" if men is not None
                 else "🔵 Meniscus start: _not set — whole ROI searched_")


if uploads:
    interactive_setup_ui(uploads)

run = st.button("Run analysis", type="primary", disabled=not uploads)

if run and uploads:
    cfg = build_config()
    if not cfg["roi"] or cfg["roi"][2] <= 0 or cfg["roi"][3] <= 0:
        st.error("ROI is not set. Set a non-zero width/height in the sidebar "
                 "(use roi_picker.py to find values for a new setup).")
        st.stop()

    work = Path(tempfile.mkdtemp(prefix="meniscus_portal_"))
    results = []
    overall = st.progress(0.0, text="Starting…")

    for i, up in enumerate(uploads):
        vid_path = work / up.name
        vid_path.write_bytes(up.getbuffer())          # uploads only; raw data untouched
        out_dir = work / "outputs" / vid_path.stem

        bar = st.progress(0.0, text=f"Analyzing {up.name}…")
        try:
            res = analyze_video(vid_path, cfg, output_dir=out_dir, verbose=False,
                                progress_cb=lambda f, b=bar, name=up.name:
                                    b.progress(f, text=f"Analyzing {name}… {int(f*100)}%"))
            bar.progress(1.0, text=f"Done: {up.name}")
            results.append((up.name, res, out_dir))
        except Exception as exc:
            bar.empty()
            st.error(f"{up.name}: {exc}")
        overall.progress((i + 1) / len(uploads), text=f"{i+1}/{len(uploads)} processed")

    overall.empty()
    st.session_state["results"] = [(n, r["df"], r["meta"], str(d)) for n, r, d in results]

# ─── Render results inline (§28.2, §28.5) ─────────────────────────────────────

results_state = st.session_state.get("results")
if results_state:
    st.divider()
    st.header("Results")

    for name, df, meta, out_dir_s in results_state:
        out_dir = Path(out_dir_s)
        outputs = sorted(out_dir.rglob("*"))
        stem = Path(name).stem

        with st.expander(f"📹 {name}", expanded=(len(results_state) == 1)):
            confidence_badge(meta)

            mcols = st.columns(4)
            mcols[0].metric("Duration", f"{meta.get('duration_s', 0):.1f} s")
            mcols[1].metric("Samples", meta.get("samples_taken", 0))
            mcols[2].metric("Scale", meta.get("scale_source", "—"))
            tot = meta.get("total_absorbed_ml")
            mcols[3].metric("Absorbed", f"{tot:.4f} mL" if tot is not None else "—")

            g1, g2 = st.columns(2)
            graph = find_output(outputs, "_graph.png")
            if graph:
                g1.image(str(graph), caption="Meniscus movement vs time", use_container_width=True)
            absg = find_output(outputs, "_absorption_ml.png")
            if absg:
                g2.image(str(absg), caption="Cumulative absorption vs time", use_container_width=True)
            sorp = find_output(outputs, "_sorptivity.png")
            if sorp:
                st.image(str(sorp), caption="Sorptivity (absorption vs √time)")

            st.subheader("Verification overlays (marks + meniscus)")
            overlays = [p for p in outputs if p.suffix.lower() == ".jpg"]
            if overlays:
                ocols = st.columns(min(3, len(overlays)))
                for j, ov in enumerate(overlays):
                    ocols[j % len(ocols)].image(str(ov), use_container_width=True,
                                                caption=ov.stem.split("_frame")[-1])

            st.subheader("Time-series data")
            st.dataframe(df, height=240, use_container_width=True)

            # Downloads (§28.5)
            st.subheader("Downloads")
            dcols = st.columns(4)
            csv = find_output(outputs, "_data.csv")
            if csv:
                dcols[0].download_button("CSV", csv.read_bytes(), file_name=csv.name,
                                         mime="text/csv")
            if graph:
                dcols[1].download_button("Movement graph", graph.read_bytes(),
                                         file_name=graph.name, mime="image/png")
            if absg:
                dcols[2].download_button("Absorption graph", absg.read_bytes(),
                                         file_name=absg.name, mime="image/png")
            dcols[3].download_button("Full package (zip)", zip_dir(out_dir),
                                     file_name=f"{stem}_outputs.zip", mime="application/zip")

    # ─── Combined / batch view (§28.8) ────────────────────────────────────────
    if len(results_state) > 1:
        st.divider()
        st.header("Combined (batch)")
        frames, summary = [], []
        for name, df, meta, _ in results_state:
            d = df.copy()
            d["video"] = name
            frames.append(d)
            summary.append({
                "video": name,
                "duration_s": round(meta.get("duration_s", 0), 2),
                "samples": meta.get("samples_taken"),
                "scale_source": meta.get("scale_source"),
                "total_absorbed_ml": meta.get("total_absorbed_ml"),
                "confidence": meta.get("confidence"),
            })
        combined = pd.concat(frames, ignore_index=True)

        st.subheader("Per-video summary")
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

        ycol = ("cumulative_absorbed_ml"
                if combined["cumulative_absorbed_ml"].notna().any()
                else ("position_mm" if combined["position_mm"].notna().any()
                      else "smoothed_displacement_px"))
        st.subheader(f"Combined: {ycol} vs time")
        chart = combined.pivot_table(index="time_s", columns="video", values=ycol)
        st.line_chart(chart)

        st.download_button("Combined CSV", combined.to_csv(index=False).encode(),
                           file_name="combined_data.csv", mime="text/csv")

st.caption("Raw input videos are never modified. Mark calibration self-scales "
           "from the tube; runs with irregular/absent marks or little movement "
           "are flagged **preliminary**.")
