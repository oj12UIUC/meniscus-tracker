# Water-Level Video Tracker (Local, On-Device)

A practical, configurable workflow for tracking the meniscus (water-level or
water-front position) in experimental videos of cementitious-material absorption
tests through tubes and pipettes. Runs entirely on your local machine. It reads
the tube's printed **graduation marks** to self-calibrate the scale, then reports
meniscus movement in **mm** and cumulative **absorbed volume in mL**, with
time-series CSV data, graphs, and verification overlays per video — plus combined
outputs across batches and a single-page **web portal**.

## What it does

For each input video, the tool:

1. Reads frames at a configurable sub-sampling rate (so 15-minute videos are practical).
2. Restricts analysis to a user-defined **ROI** (region of the tube/pipette only).
3. Reduces the ROI to a 1-D intensity profile along the tube axis.
4. Detects the **meniscus** (water-level / advancing front) on each sampled frame
   using a **gradient peak**, a **thresholded segment**, or a **frame-difference** rule.
5. Detects the tube's **graduation marks** and measures their pixel spacing, then
   uses the stated value of one interval (mm or mL) to convert meniscus movement
   into physical units and cumulative absorbed volume — self-calibrating from the
   tube (PRD §27). Falls back to a manual `pixels_per_cm` if marks can't be read.
6. Tracks position over the **full duration**, using real FPS for time alignment.
7. Writes a per-video CSV, movement/absorption graphs, metadata, and labeled
   overlay frames (marks + meniscus) so you can verify the detection is on the
   actual boundary — not tape, labels, clamps, or reflections.
8. Flags each run **strong** or **preliminary** based on mark regularity and
   detected movement (PRD §16), so uncertain results aren't overstated.

## Files in this package

| File | Purpose |
|---|---|
| `water_tracker.py` | Core engine: ROI extraction, meniscus + mark detection, mm/mL conversion, output writing |
| `graduation.py` | Graduation-mark detection — finds the tube's tick marks and their spacing |
| `config.py` | Per-setup profiles (Burgess / CEMEX / Holcim vertical, C202 / NP / Fly_Ash horizontal, general) |
| `roi_picker.py` | Interactive ROI selector — draw a rectangle to get x/y/w/h |
| `calibrate.py` | Manual scale calibration — click a known span to set mm/mL per pixel by hand |
| `analyze_single.py` | Run the tracker on one video from the command line |
| `batch_runner.py` | Recurse a folder, process every video, write combined outputs |
| `app.py` | Streamlit **web portal** — upload, analyze, and view results on one page |
| `requirements.txt` | Python dependencies |
| `METHODOLOGY.md` | Step-by-step explanation of how the tracking works |
| `PROJECT_SUMMARY.md` | One-page research share-out |

## Installation (one-time)

From this folder, in PowerShell:

```powershell
pip install -r requirements.txt
```

That installs OpenCV, NumPy, pandas, matplotlib, and SciPy.

## Quick start

### Step 1 — pick an ROI

The ROI is the rectangle around the *tube interior* only (no tape, no clamps, no
labels, no background). The defaults in `config.py` are placeholders — calibrate
once per setup type by running:

```powershell
python roi_picker.py "C:\Users\oorji\Box\ER_2\Raw Videos ER2\Burgess\Burg_T1.MOV"
```

Drag a rectangle, press **ENTER**. The original-resolution `(x, y, w, h)` are
printed — paste into `config.py` under the matching profile (e.g. `burgess_vertical`).

### Step 2 — analyze a single video

```powershell
python analyze_single.py --video "C:\Users\oorji\Box\ER_2\Raw Videos ER2\Burgess\Burg_T1.MOV" --profile burgess_vertical
```

Outputs land in `.\outputs\Burg_T1\`:

```
Burg_T1_data.csv          time, position, displacement, marks, position_mm, absorbed_ml
Burg_T1_graph.png         time vs. meniscus movement (mm when calibrated, else px)
Burg_T1_absorption_ml.png time vs. cumulative absorbed volume (when mL available)
Burg_T1_sorptivity.png    absorption vs. √time with fitted slope (if enabled)
Burg_T1_meta.json         video + run metadata, scale source, confidence
verification\
    Burg_T1_frame000000.jpg   (ROI + detected marks + meniscus + confidence label)
    Burg_T1_frame001234.jpg
    ...
```

**Always open the verification frames first** and confirm the red line really
sits on the meniscus (and the gray lines on the printed marks) before trusting
the CSV. The run's **strong / preliminary** flag is printed and stored in the
metadata.

### Step 3 — batch a folder

```powershell
python batch_runner.py --folder "C:\Users\oorji\Box\ER_2\Raw Videos ER2"
```

The runner walks every subfolder, infers the profile from the folder name
(`Burgess` → `burgess_vertical`, etc.), processes every video, and writes:

```
outputs\Raw Videos ER2\
    Burgess\Burg_T1\Burg_T1_data.csv
    Burgess\Burg_T1\Burg_T1_graph.png
    ...
    combined_data.csv
    combined_graph_Burgess.png        (per-folder movement, one series per video)
    combined_absorption_Burgess.png   (per-folder absorption in mL / movement in mm)
    combined_graph_Fly_Ash.png
    combined_absorption_Fly_Ash.png
    ...
    batch_summary.csv                 (per video: scale source, total absorbed mL, confidence)
```

For a quick smoke test, restrict to a few videos:

```powershell
python batch_runner.py --folder "C:\Users\oorji\Box\ER_2\Raw Videos ER2" --max 3
```

The tool reads the raw videos directly — it does **not** modify or move them.

### Step 4 — web portal (upload + view on one page)

For a no-scripting workflow, launch the single-page portal:

```powershell
streamlit run app.py
```

It opens in your browser. Upload one or more videos (drag and drop), pick a setup
profile (and tweak the ROI / detection / mark-calibration settings in the sidebar
if needed), click **Run analysis**, and the movement graph, absorption graph,
time-series table, verification overlays, and confidence label render on the same
page — with buttons to download the CSV, graphs, and a full output zip. Uploading
multiple files adds a combined batch view.

The portal calls the same engine as the CLIs (no duplicated logic). Very large
multi-GB clips (e.g. 20–40 min `.MOV`) are better run through `batch_runner.py`
than uploaded through the browser.

### Optional CLI overrides

```powershell
# Override ROI for one-off runs without editing config.py
python analyze_single.py --video v.mp4 --profile general --roi 200 50 200 600 --orientation vertical

# Process every 5th frame for higher time resolution (more samples, slower)
python batch_runner.py --folder . --sample-n 5

# Force a profile globally instead of auto-detecting
python batch_runner.py --folder . --no-auto-profile --profile burgess_vertical
```

## Configurable parameters (per profile in `config.py`)

| Key | Meaning |
|---|---|
| `orientation` | `"vertical"` for rising water, `"horizontal"` for advancing front |
| `roi` | `(x, y, width, height)` — tube interior only |
| `feature_darker` | `True` if water is darker than the surrounding background |
| `edge_type` | `"gradient"` (derivative peak), `"threshold"` (cut at mean), or `"frame_diff"` (change vs first frame — robust to fixed markings/tape) |
| `use_first_edge` | `True` = first qualifying edge along axis, `False` = last (useful for an advancing front) |
| `search_start` / `search_end` | Pixel bounds along the tube axis inside the ROI |
| `threshold_bias` | Offset added to the auto-computed threshold (sensitivity tuning) |
| `smoothing_window` | Savitzky-Golay window size for the output curve, odd integer, `0` = off |
| `sample_every_n_frames` | Sub-sampling: process 1 frame per N (15 @ 30fps ≈ 0.5s spacing) |
| `mark_calibration` | `True` to scale from the tube's graduation marks (recommended) |
| `mark_interval_value` | Physical value of **one** mark interval (e.g. `1.0`) |
| `mark_interval_unit` | `"mm"` (length graduations) or `"mL"` (volume graduations) |
| `marks_darker` | `True` if the printed ticks are darker than the tube |
| `tube_diameter_mm` | Inner diameter — only needed to convert **mm** marks into mL |
| `absorption_direction` | `"auto"`, `"increasing"`, or `"decreasing"` — keeps absorbed volume positive (PRD §27.5) |
| `sorptivity` | `{"enabled", "t_min_s", "t_max_s"}` — optional absorption-vs-√time fit (ASTM C1585 style) |
| `manual_calibration` | `True` to force a hand-measured scale (overrides auto marks) |
| `manual_span_px` / `manual_span_value` / `manual_span_unit` | A known span: this many pixels = this value in mm or mL |
| `pixels_per_cm` | Last-resort fallback used only when neither manual nor marks are set |

## Units & calibration (graduation marks first)

The tube carries printed tick marks at a regular interval. When
`mark_calibration` is on, the engine detects those marks inside the ROI, measures
their average pixel spacing, and combines it with the stated value of one interval
to produce a per-video conversion factor — read from the tube, not assumed. This
self-calibrates against camera distance, zoom, and minor perspective changes.

- If the marks are **length** graduations (`mark_interval_unit: "mm"`), the CSV
  reports `position_mm`; absorbed mL also requires `tube_diameter_mm`.
- If the marks are **volume** graduations (`mark_interval_unit: "mL"`), absorbed
  `cumulative_absorbed_ml` is reported directly — no diameter needed.
- These interval values are **starting points** — confirm them against a
  verification overlay (which now draws the detected marks) for each setup.

If marks are missing, too few, or irregular, the run is flagged **preliminary**
and falls back to `pixels_per_cm` (measure a known length in pixels and divide by
cm), or to raw pixels if neither is available.

### Manual calibration (the failsafe — recommended when marks are faint)

On many tubes the printed ticks are too faint or uneven to auto-detect reliably.
In that case, calibrate the scale by hand once and the tool uses that exact value
— this **takes precedence over** automatic mark detection.

```powershell
# Crop+zoom to the tube so the faint marks are visible, then click the two
# ends of a span you can read (e.g. the 0 mark and the 5 mark).
python calibrate.py "C:\path\to\video.MOV" --roi 339 512 1347 97
```

It asks what that span represents (e.g. `5` `mm`, or `0.05` `mL`) and prints a
snippet to paste into the matching profile in `config.py`:

```python
"manual_calibration": True,
"manual_span_px":     520.0,
"manual_span_value":  5,
"manual_span_unit":   "mm",
```

…or pass it per run without editing config:

```powershell
python analyze_single.py --video "v.MOV" --profile fly_ash_horizontal `
    --manual-span-px 520 --manual-span-value 5 --manual-span-unit mm
```

In the **web portal**, the *“🎯 Interactive setup”* panel lets you do the whole
setup by clicking on a preview frame (needs `streamlit-image-coordinates`, in
`requirements.txt`):

- **① Draw ROI box** — click two opposite corners to place/resize the green
  measurement box on the tube.
- **② Calibrate scale** — click the two ends of a known span (e.g. the 0 and 5
  marks), then type what it represents (mm or mL). Overrides auto marks.
- **③ Mark meniscus start** — click where the water starts; a blue line is drawn
  and detection is restricted to from there onward (so labels/clutter above the
  start are ignored).

Use the **Zoom** slider so the faint marks are easy to click, and the **Reset**
buttons to redo any of the three.

Scale precedence: **manual (forced) → detected marks → `pixels_per_cm` → raw pixels.**

## Honest caveats

- Tracking quality depends on the **visibility of the actual meniscus** and the
  **clarity of the graduation marks**. Strong contrast + clean marks → credible
  mm/mL curves flagged **strong**. Setups dominated by tape, labels, reflections,
  weak movement, or irregular marks are flagged **preliminary** — always check the
  verification overlays (marks + meniscus are both drawn).
- ROI is per-setup-type at minimum, and may need to be per-video if framing
  differs between recordings of the same material.
- A single universal detection rule does not work across every video. The
  profile system exists exactly for this reason; expect to tune per category.
- `CEMEX` and `Holcim` profiles are new placeholders — set their ROI (via
  `roi_picker.py`) and confirm the mark interval before trusting their numbers.

See `METHODOLOGY.md` for the step-by-step explanation and `PROJECT_SUMMARY.md`
for the share-out version.
