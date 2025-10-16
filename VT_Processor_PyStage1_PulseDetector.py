#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
NeuroEng Voltage Transient Processor - Pulse Detector
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-Aug-29
Updated:    2025-Oct-15 
Version:    v3.3

DESCRIPTION
---------------------------
- Pick one or more TEK *.CSV files (TIME, CH1, CH2 with header ~row16).
- Auto-detect the biphasic pulse on CURRENT, trim with padding, and
  shift time so the selected window starts at -0.1 ms.
- Plot Voltage (top) and Current (bottom)
- Figure title = TEK file name.

Notes for editing:
- Flip which channel is voltage/current by changing `dict_TEK_CHANNEL_MAP`.
- Adjust label PULSE_SHIFT with `YLABEL_X` or use just `axes.labelpad`.
"""

# =============================================================================
# 📦 IMPORT REQUIRED PACKAGES
# =============================================================================
print("_"*80 + "")
print("🟠 📦 Importing Packages...")

import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from datetime import datetime
from tkinter import Tk, filedialog

import platform
import getpass
import json
import pickle

# =============================================================================
# 🛠️ GLOBAL CONFIGURATION
# =============================================================================
print("🟠 🛠️ Global Configuration...")

# =========================
# Script Info
# =========================

script_name = "VT_Plotter_Batch_PulseDetector_v3.3.py"
script_version = "2025-09-03"
dict_SYSTEM_INFO = {
    "system": platform.system(),
    "release": platform.release(),
    "python_version": platform.python_version(),
    "user": getpass.getuser()
}

# =========================
# 🛠️ 📦 Default Channel Map
# =========================
# ---- Channel → Meaning mapping (edit if CH1/CH2 are swapped) ----
dict_TEK_CHANNEL_MAP = {
    "time":    "TIME",
    "voltage": "CH1",   # switch to "CH2" if voltage is on CH2
    "current": "CH2",   # switch to "CH1" if current is on CH1
}

# Canonical column names (after standardization)
xcol_t = "Time (µs)"
ycol_v = "Voltage (V)"
ycol_i = "Current (A)"

# Axis labels (for plots)
dict_AXIS_LABELS = {
    "time":    "Time (µs)",
    "voltage": "Voltage (V)",
    "current": "Current (A)",
}

# =========================
# Global Defaults & style
# =========================

# ---- Channel → Meaning mapping (edit if CH1/CH2 are swapped) ----
dict_TEK_CHANNEL_MAP = {
    "time":    "TIME",
    "voltage": "CH1",    # set to "CH2" if voltage is on CH2
    "current": "CH2",    # set to "CH1" if current is on CH1
}

# Canonical column names (after standardization)
xcol_t = "Time (µs)"
ycol_v = "Voltage (V)"
ycol_i = "Current (A)"

# ---- Axis labels (for plots) ----
# This will be displayed in the plot
dict_AXIS_LABELS = {
    "time":    "Time (µs)",
    "voltage": "Voltage (V)",
    "current": "Current (A)",
}

# =========================
# 🎨 Global Plot Style (matplotlib.rcParams)
# =========================
# ---- (fonts, grid, legend, etc.) ----

rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.labelpad": 4,           # distance from tick labels
    "yaxis.labellocation": "center",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.linewidth": 1,
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1,
    "figure.figsize": (7.2, 4.0),
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "savefig.dpi": 300,
    "legend.title_fontsize": 10,
    "legend.fontsize": 9,
    "legend.loc": "upper left",
    "legend.frameon": False,
})

# ---- Consistent line styles (matches your green/pink look) ----
dict_STYLE_V_raw = dict(color="#E6B800", linewidth=1.0, alpha=0.5, zorder=1)  # Light Amber
dict_STYLE_V_sel = dict(color="#E6B800", linewidth=1.2, alpha=1.0, zorder=3)  # Amber
dict_STYLE_I_raw = dict(color="#1E90FF", linewidth=1.0, alpha=0.5, zorder=1)  # Light Dodger Blue
dict_STYLE_I_sel = dict(color="#1E90FF", linewidth=1.2, alpha=1.0, zorder=3)  # Dodger Blue

# =========================
# Axis Parameters (µs)
# =========================
XRANGE_EXTRA_US = 100   # extra margin beyond the detected window on each side (try 100 or 200)
X_MAJOR_TICK_US   = 100   # major tick spacing in µs
X_MINOR_PER_MAJOR = 2     # number of minor intervals between majors (2 => 50 µs)
YLABEL_X  = -0.01 # y-label x-position in axes coords; negative pushes left of spine
YLABEL_Y   = 0.50  # y-label vertical position in axes coords (0–1)
_AUTO_YLABEL_ADJUST = False

def _adjust_all_figs_ylabel_position(x=YLABEL_X, y=YLABEL_Y):
    """Shift y-axis labels for all axes in all open figures."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        for ax in fig.get_axes():
            if hasattr(ax, "yaxis"):
                ax.yaxis.set_label_coords(x, y)

# Monkey-patch plt.show so labels are nudged AFTER layout is finalized
_original_show = plt.show
def _patched_show(*args, **kwargs):
    plt.draw()  # finalize layout (tight/constrained)
    if _AUTO_YLABEL_ADJUST:
        _adjust_all_figs_ylabel_position()
    return _original_show(*args, **kwargs)
plt.show = _patched_show


# ---- [Helper Function] Axis Labels ----

def _place_ylabel_left(ax, x=None, y=None):
    """Put y-label left of the spine using axes coords (no autosnap)."""
    if x is None: x = YLABEL_X
    if y is None: y = YLABEL_Y
    lab = ax.yaxis.get_label()
    lab.set_transform(ax.transAxes)   # use axes coordinates
    lab.set_ha("right"); lab.set_va("center")
    lab.set_clip_on(False)            # allow outside drawing
    lab.set_position((x, y))          # same as set_label_coords(x, y)

def label_axes(ax_v, ax_i):
    ax_v.set_ylabel(dict_AXIS_LABELS["voltage"])
    ax_i.set_ylabel(dict_AXIS_LABELS["current"])
    ax_i.set_xlabel(dict_AXIS_LABELS["time"])
    _place_ylabel_left(ax_v)
    _place_ylabel_left(ax_i)

# =============================================================================
# 📈 PULSE DETECTOR
# =============================================================================

# =========================
# Pulse Detection Parameters (µs)
# =========================
MIN_USEFUL_WIDTH_US     = 40       # minimum lobe width
NOISE_MULT              = 6.0      # MAD multiplier
FALLBACK_THRESH_A       = 5e-6     # floor for threshold (A) → stays in Amps, not time
FRAC_OF_PEAK            = 0.07     # at least 7% of p99(|I|)
MIN_LOBE_WIDTH_US       = 20       # 20 µs
AMP_SYMMETRY_BIAS       = 0.25
MAX_HOLE_WITHIN_US      = 60       # 60 µs
MAX_INTERPHASE_GAP_US   = 500      # 500 µs
PULSE_PAD_US            = 200      # ± µs padding
PULSE_START_US          = 0        # Desired trace start time (µs)
PULSE_SHIFT_US          = PULSE_START_US - PULSE_PAD_US

# --- Internal conversion: µs → seconds
MIN_USEFUL_WIDTH        = MIN_USEFUL_WIDTH_US * 1e-6
MIN_LOBE_WIDTH          = MIN_LOBE_WIDTH_US * 1e-6
MAX_HOLE_WITHIN_LOBE    = MAX_HOLE_WITHIN_US * 1e-6
MAX_INTERPHASE_GAP      = MAX_INTERPHASE_GAP_US * 1e-6
PULSE_PAD               = PULSE_PAD_US * 1e-6
PULSE_START             = PULSE_START_US * 1e-6
PULSE_SHIFT             = PULSE_SHIFT_US * 1e-6

# =========================
# Helpers (Biphasic Pulse Detector)
# =========================
def _mad(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def _adaptive_threshold(i):
    abs_i = np.abs(i)
    cutoff = np.quantile(abs_i, 0.5)
    pool = i[abs_i <= cutoff]
    sigma = _mad(pool) if pool.size >= 20 else _mad(i)
    thr_noise = NOISE_MULT * sigma
    p99 = np.percentile(abs_i, 99)
    thr_peak = FRAC_OF_PEAK * p99
    return max(FALLBACK_THRESH_A, thr_noise, thr_peak)

def _contiguous_regions(mask):
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    jumps = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[jumps + 1]]
    ends   = np.r_[idx[jumps], idx[-1]]
    return list(zip(starts, ends))

def _merge_same_sign_lobes(lobes, t, max_hole):
    if not lobes:
        return []
    merged = [lobes[0].copy()]
    for L in lobes[1:]:
        prev = merged[-1]
        gap = t[int(L["s"])] - t[int(prev["e"])]
        if (L["sign"] == prev["sign"]) and (0 <= gap <= max_hole):
            prev["e"] = int(L["e"])
            prev["width"] = t[prev["e"]] - t[prev["s"]]
            prev["amp"] = max(prev["amp"], L["amp"])
        else:
            merged.append(L.copy())
    return merged

def _find_biphasic_window(t, i):
    """Return (i0, i1, thr) indices covering the biphasic window, with padding."""
    if len(t) < 10:
        return None
    thr = _adaptive_threshold(i)
    mask = np.abs(i) > thr
    runs = _contiguous_regions(mask)
    if not runs:
        return None

    # summarize lobes
    lobes = []
    for s, e in runs:
        s = int(s); e = int(e)
        width = t[e] - t[s]
        if width <= 0:
            continue
        seg = i[s:e+1]
        amp = float(np.median(np.abs(seg)))
        sign = np.sign(np.median(seg)) or 1.0
        lobes.append({"s": s, "e": e, "amp": amp, "sign": float(sign), "width": width})
    if not lobes:
        return None

    # bridge small holes and filter
    lobes = _merge_same_sign_lobes(lobes, t, MAX_HOLE_WITHIN_LOBE)
    lobes = [L for L in lobes if L["width"] >= MIN_LOBE_WIDTH]
    if not lobes:
        return None

    # pair opposite-signed neighbors
    best_score = -np.inf; s_pair = e_pair = None
    for k in range(len(lobes) - 1):
        L1, L2 = lobes[k], lobes[k+1]
        if L1["sign"] * L2["sign"] >= 0:
            continue
        gap = t[int(L2["s"])] - t[int(L1["e"])]
        if gap < 0 or gap > MAX_INTERPHASE_GAP:
            continue
        sym = min(L1["amp"], L2["amp"]) / max(L1["amp"], L2["amp"])
        score = (L1["amp"] + L2["amp"]) * (L1["width"] + L2["width"]) * (AMP_SYMMETRY_BIAS + (1 - AMP_SYMMETRY_BIAS) * sym)
        if score > best_score:
            best_score = score; s_pair, e_pair = int(L1["s"]), int(L2["e"])

    if best_score > -np.inf:
        start_t = max(t[0], t[s_pair] - PULSE_PAD)
        end_t   = min(t[-1], t[e_pair] + PULSE_PAD)
        i0 = int(np.searchsorted(t, start_t, side="left"))
        i1 = int(np.searchsorted(t, end_t,   side="right") - 1)
        i0 = max(0, min(i0, len(t)-1)); i1 = max(0, min(i1, len(t)-1))
        if i1 > i0:
            return i0, i1, thr
    return None

# =========================
# Global Data Dictionaries
# =========================
dict_CSV_RawTimeµs = {}      # original unmodified TEK CSV
dict_CSV_Detected = {}  # detected biphasic window (not shifted)
dict_CSV_AutoOffset = {}   # detected window with time offset applied

# =========================
# Export settings
# =========================
SAVE_FORMATS = ("png")   # add more if you like, e.g., "eps"
RC_SUBSET_KEYS = [
    "font.family","font.size","axes.labelsize","axes.titlepad","axes.titlesize",
    "axes.linewidth","axes.facecolor","axes.grid","grid.color","grid.linestyle",
    "grid.linewidth","xtick.labelsize","ytick.labelsize","lines.linewidth",
    "figure.figsize","savefig.dpi","legend.fontsize","legend.loc"
]

# =========================
# Export helpers
# =========================
def _safe_open(path_to_open):
    try:
        if platform.system() == "Windows":
            os.startfile(path_to_open)
        elif platform.system() == "Darwin":
            os.system(f"open '{path_to_open}'")
        else:
            os.system(f"xdg-open '{path_to_open}'")
    except Exception as e:
        print(f"⚠️ Could not auto-open: {e}")

# =============================================================================
# 📤 [FUNCTION] EXPORT / SAVE
# =============================================================================
def save_all_outputs(
    *,
    dict_CSV_RawTimeµs: dict,
    dict_CSV_Detected: dict,
    dict_CSV_AutoOffset: dict,
    plot_specs: list,                 # list of {"fig": fig, "ax_v": ax_v, "ax_i": ax_i, "plot_file_name_base": str}
    folder_path_parent: str,
    folder_path_output_main: str,
    folder_path_output_dated: str,
    timestamp_str: str,
    save_formats: tuple = SAVE_FORMATS,
    script_name: str = script_name,
    script_version: str = script_version,
    system_info: dict = dict_SYSTEM_INFO,
    channel_map: dict = dict_TEK_CHANNEL_MAP,
    axis_labels: dict = dict_AXIS_LABELS,
    rc_subset_keys: list = RC_SUBSET_KEYS,
    pulse_params: dict = None,
    auto_open: bool = True
):
    """
    Saves:
      • per-file plots (.png)
      • RawTimeµs/Detected/AutoOffset tables (.CSV)
      • DataFrame dictionaries (.pkl): RawTimeµs, Detected, AutoOffset
      • PyPD_Summary (.json + .txt)
    """

    print("\n🟠 📤 Exporting files...")

    os.makedirs(folder_path_output_main, exist_ok=True)
    os.makedirs(folder_path_output_dated, exist_ok=True)

    # ---------------------------
    # 🖼️ Save per-file plots
    # ---------------------------
    list_saved_plot_files = []
    for spec in plot_specs:
        fig       = spec["fig"]
        plot_file_name_base = spec["plot_file_name_base"]  
        # e.g., f"Plot_{file_name}_AutoOffset"
        
        for ext in save_formats:
            out_path = os.path.join(folder_path_output_dated, f"{plot_file_name_base}.{ext}")
            fig.savefig(out_path, dpi=rcParams.get("savefig.dpi", 300))
            list_saved_plot_files.append(os.path.basename(out_path))
            print(f"✅ 🖼️ Saved plot: {os.path.basename(out_path)}")

    # ---------------------------
    # 💾 Save tables (.CSV uppercase)
    # ---------------------------
    for process_tag, dct in (("RawTimeµs", dict_CSV_RawTimeµs), ("Detected", dict_CSV_Detected), ("AutoOffset", dict_CSV_AutoOffset)):
        for fname, df_out in dct.items():
            out_path = os.path.join(folder_path_output_dated, f"PyPD_{fname}_{process_tag}.CSV")
            df_out.to_csv(out_path, index=False)
    print("✅ 💾 Saved TEK.CSV tables → RawTimeµs/Detected/AutoOffset")
    
    # ---------------------------
    # 📚 Save dictionaries (.pkl)
    # ---------------------------
    pkl_raw_path      = os.path.join(folder_path_output_dated, "PyPD_Dictionary_AllDataFiles_wRawTimeµs.pkl")
    pkl_detected_path = os.path.join(folder_path_output_dated, "PyPD_Dictionary_AllDataFiles_wDetected.pkl")
    pkl_offset_path   = os.path.join(folder_path_output_dated, "PyPD_Dictionary_AllDataFiles_wAutoOffset.pkl")

    with open(pkl_raw_path, "wb") as f:
        pickle.dump(dict_CSV_RawTimeµs, f)
    with open(pkl_detected_path, "wb") as f:
        pickle.dump(dict_CSV_Detected, f)
    with open(pkl_offset_path, "wb") as f:
        pickle.dump(dict_CSV_AutoOffset, f)
    print("✅ 📚 Saved pickles: DFdictionary_TEK_raw/Detected/AutoOffset.pkl")

    # ---------------------------
    # 📝 PyPD_Summary (.json + .txt)
    # ---------------------------
    rcparams_subset = {k: rcParams[k] for k in rc_subset_keys if k in rcParams}

    if pulse_params is None:
        pulse_params = {
            "MIN_USEFUL_WIDTH_US": MIN_USEFUL_WIDTH_US,
            "NOISE_MULT": NOISE_MULT,
            "FALLBACK_THRESH_A": FALLBACK_THRESH_A,
            "FRAC_OF_PEAK": FRAC_OF_PEAK,
            "MIN_LOBE_WIDTH_US": MIN_LOBE_WIDTH_US,
            "AMP_SYMMETRY_BIAS": AMP_SYMMETRY_BIAS,
            "MAX_HOLE_WITHIN_US": MAX_HOLE_WITHIN_US,
            "MAX_INTERPHASE_GAP_US": MAX_INTERPHASE_GAP_US,
            "PULSE_PAD_US": PULSE_PAD_US,
            "PULSE_START_US": PULSE_START_US,
            "PULSE_SHIFT_US": PULSE_SHIFT_US,
            "XRANGE_EXTRA_US": XRANGE_EXTRA_US,
            "X_MAJOR_TICK_US": X_MAJOR_TICK_US,
            "X_MINOR_PER_MAJOR": X_MINOR_PER_MAJOR,
        }

    # Compute offset deltas (detected start vs offset start) in µs
    offset_summary_us = {}
    for fname, df_off in dict_CSV_AutoOffset.items():
        t_off_start = float(df_off["time_us"].iloc[0])
        t_det_start = float(dict_CSV_Detected[fname]["time_us"].iloc[0])
        offset_summary_us[fname] = {
            "detected_start_us": t_det_start,
            "offset_start_us": t_off_start,
            "delta_us": t_off_start - t_det_start
        }

    tables_saved = [f"{k}_{process_tag}.CSV"
                    for k in dict_CSV_RawTimeµs.keys()
                    for process_tag in ("RawTimeµs", "Detected", "AutoOffset")]

    PyPD_Summary = {
        "timestamp": timestamp_str,
        "script": {"name": script_name, "version": script_version},
        "system": system_info,
        "input_folder": folder_path_parent,
        "output_folder": folder_path_output_dated,
        "n_files_processed": len(dict_CSV_RawTimeµs),
        "file_keys": list(dict_CSV_RawTimeµs.keys()),
        "channel_map": channel_map,
        "axis_labels": axis_labels,
        "rcparams_subset": rcparams_subset,
        "pulse_params": pulse_params,
        "offset_summary_us": offset_summary_us,
        "plots_saved": list_saved_plot_files,
        "tables_saved": tables_saved,
        "pickles_saved": [
            os.path.basename(pkl_raw_path),
            os.path.basename(pkl_detected_path),
            os.path.basename(pkl_offset_path),
        ],
    }

    json_path = os.path.join(folder_path_output_dated, "PyPD_Summary.json")
    with open(json_path, "w") as f:
        json.dump(PyPD_Summary, f, indent=2)
    print("✅ 📝 Saved PyPD_Summary JSON: PyPD_Summary.json")

    txt_path = os.path.join(folder_path_output_dated, "PyPD_Summary.txt")
    with open(txt_path, "w") as f:
        f.write("Voltage Transient Batch Processing — PyPD_Summary.\n")
        f.write("====================================================\n\n")
        f.write(f"Timestamp:      {timestamp_str}\n")
        f.write(f"Script:         {script_name} (v{script_version})\n")
        f.write(f"User/System:    {system_info['user']} | "
                f"{system_info['system']} {system_info['release']} | "
                f"Python {system_info['python_version']}\n\n")
        f.write(f"Input Folder:   {folder_path_parent}\n")
        f.write(f"Output Folder:  {folder_path_output_dated}\n")
        f.write(f"Files Processed: {len(dict_CSV_RawTimeµs)}\n\n")

        f.write("Channel Map:\n")
        for k, v in channel_map.items(): f.write(f"  - {k}: {v}\n")
        f.write("\nAxis Labels:\n")
        for k, v in axis_labels.items(): f.write(f"  - {k}: {v}\n")

        f.write("\nPulse Detector Parameters:\n")
        for k, v in pulse_params.items(): f.write(f"  - {k}: {v}\n")

        f.write("\nPer-file AutoOffset Summary (µs):\n")
        for k, vals in offset_summary_us.items():
            f.write(f"  - {k}: detected_start={vals['detected_start_us']:.3f}, "
                    f"offset_start={vals['offset_start_us']:.3f}, "
                    f"delta={vals['delta_us']:.3f}\n")

        f.write("\nPlots Saved:\n")
        for p in list_saved_plot_files: f.write(f"  - {p}\n")

        f.write("\nTables Saved (.CSV):\n")
        for name in tables_saved: f.write(f"  - {name}\n")

        f.write("\nPickles Saved:\n")
        for p in PyPD_Summary["pickles_saved"]: f.write(f"  - {p}\n")

        f.write("\nrcParams Subset:\n")
        for k, v in rcparams_subset.items(): f.write(f"  - {k}: {v}\n")

    print("✅ 🗒️ Saved PyPD_Summary TXT: VT_PyPD_Summary.txt")

    if auto_open:
        _safe_open(folder_path_output_dated)

    return {
        "output_folder": folder_path_output_dated,
        "json": json_path,
        "txt": txt_path,
        "plots": list_saved_plot_files,
        "tables": tables_saved,
        "pickles": PyPD_Summary["pickles_saved"],
    }

# =============================================================================
# 🔄 [FUNCTION] CSV Loader + 📊 Plotter
# =============================================================================

def load_tek_csv_files(file_path=None, skip_rows=15):
    """Pick files (if not provided), detect window, plot Voltage/Current."""
    
    # =========================
    # SELECT FILES, ORGANIZE FOLDERS
    # =========================
    
    #--------------------
    # 📁 FILE PICKER: Choose CSVs if not provided
    #--------------------
    if not file_path:
        print("🟠 🖥️ Opening File Picker")
        print("\n❗ ACTION REQUIRED ❗")
        print("👉 📁 Select TEK.CSV files")
        Tk().withdraw()
        file_path = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not file_path:
            print("\n🚫 No files selected. Exiting.")
            print("_" * 80)
            sys.exit(0)

    #--------------------
    # 📁 IMPORT FOLDER
    #--------------------
    folder_path_parent = os.path.dirname(file_path[0])
    folder_name_parent = os.path.basename(folder_path_parent)
    print(f"✅ Selected folder: → {folder_path_parent}")
    print("✅ Selected files:")
    for i, path in enumerate(file_path, start=1):
        print(f"      {i}. {os.path.basename(path)}")
        
    # ---------------------------
    # 📤 📂 CREATE EXPORT FOLDERS
    # ---------------------------
    # Timestamp for folder naming
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Main output folder
    folder_path_output_main = os.path.join(folder_path_parent, "VT_Py_Outputs")
    folder_name_output_main = os.path.basename(folder_path_output_main)
    
    # Subfolder with timestamp
    folder_path_output_dated = os.path.join(folder_path_output_main, f"VT_Py_Outputs_PulseDetector_{timestamp_str}")
    folder_name_output_dated = os.path.basename(folder_path_output_dated)
    
    # Create the folders
    os.makedirs(folder_path_output_main, exist_ok=True)
    os.makedirs(folder_path_output_dated, exist_ok=True)
    
    # Track plots to save after processing
    plot_specs = []

    # Track what we save per file (for the TXT summary)
    list_saved_plot_files = []

    print(f"\n✅ 📂 Created Export Folder: {folder_name_parent} → {folder_name_output_main} → {folder_name_output_dated}")
    
    # =========================
    # PROCESS EACH CSV FILE
    # =========================

    for csv_path in file_path:
        file_name = os.path.splitext(os.path.basename(csv_path))[0]

        # 1) Read CSV (header around row 16)
        df = pd.read_csv(csv_path, skiprows=skip_rows, engine="python", sep=None)

        # Ensure required columns exist
        required = {"TIME", "CH1", "CH2"}
        if not required.issubset(df.columns):
            print(f"❌ {file_name}: missing required columns {required}. Skipping.")
            continue

        # 2) Convert to numeric + drop rows with NaNs in key columns
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=list(required)).copy()

        # 3) Neutral arrays via mapping
        t_sec = df[dict_TEK_CHANNEL_MAP["time"]].to_numpy()
        v_dat = df[dict_TEK_CHANNEL_MAP["voltage"]].to_numpy()
        i_dat = df[dict_TEK_CHANNEL_MAP["current"]].to_numpy()

        # 4) Auto-detect biphasic window on CURRENT
        found = _find_biphasic_window(t_sec, i_dat)
        if found is None:
            print(f"❌ Skipping {file_name} plot: No biphasic pulse found.")
            continue
        i0, i1, thr = found
        if i1 <= i0:
            print(f"❌ {file_name}: invalid window indices ({i0}, {i1}).")
            continue

        # 5) Trim + align so selected starts at user defined "PULSE_START_US" µs
        df_sel = df.iloc[i0:i1+1].copy()
        t_start_sel_s = df_sel[dict_TEK_CHANNEL_MAP["time"]].iloc[0]      # seconds
        t_shift = (PULSE_SHIFT) - t_start_sel_s                    # this shifts the traces in time
  
        # 6) Build shifted time in µs (both full & detected)
        t_raw_us = (df[dict_TEK_CHANNEL_MAP["time"]] + t_shift) * 1e6
        t_sel_us = (df_sel[dict_TEK_CHANNEL_MAP["time"]] + t_shift) * 1e6

        v_sel = df_sel[dict_TEK_CHANNEL_MAP["voltage"]].to_numpy()
        i_sel = df_sel[dict_TEK_CHANNEL_MAP["current"]].to_numpy()
        
        # 6.1) Build DataFrames for the three dictionaries
        
        # RawTimeµs (use original TEK time; no trimming; no shifting)
        df_out_raw = pd.DataFrame({
            "time_us": df[dict_TEK_CHANNEL_MAP["time"]] * 1e6,     # original TEK TIME in µs
            "voltage_V": df[dict_TEK_CHANNEL_MAP["voltage"]],
            "current_A": df[dict_TEK_CHANNEL_MAP["current"]],
        })
        
        # Detected (selected biphasic window; not shifted)
        df_out_detected = pd.DataFrame({
            "time_us": df_sel[dict_TEK_CHANNEL_MAP["time"]] * 1e6, # original TEK TIME in µs (unshifted)
            "voltage_V": v_sel,
            "current_A": i_sel,
        })
        
        # AutoOffset (same detected window, but shifted so start aligns to DESIRED_START)
        df_out_offset = pd.DataFrame({
            "time_us": t_sel_us,   # shifted time in µs
            "voltage_V": v_sel,
            "current_A": i_sel,
        })
        
        # Store into dictionaries
        dict_CSV_RawTimeµs[file_name] = df_out_raw
        dict_CSV_Detected[file_name] = df_out_detected
        dict_CSV_AutoOffset[file_name] = df_out_offset

        # 7) Plot (Voltage top, Current bottom)
        fig, (ax_v, ax_i) = plt.subplots(
            2, 
            1, 
            sharex=True,
            figsize=(rcParams["figure.figsize"][0], rcParams["figure.figsize"][1] * 1.2)
        )
        
        #Figure Title
        fig.suptitle(f"{folder_name_parent}_$\\bf{{{file_name}}}$")
        
        # Plot - Raw (x-axis, y-axis, label, style)
        ax_v.plot(t_raw_us, v_dat, label="Raw",   **dict_STYLE_V_raw)
        ax_i.plot(t_raw_us, i_dat, label="Raw",   **dict_STYLE_I_raw)

        # Plot - Detected+Shifted 
        ax_v.plot(t_sel_us, v_sel, label=f"Detected (Pad ±{PULSE_PAD_US} µs), Aligned (Start {PULSE_START_US:.0f} µs)", **dict_STYLE_V_sel)
        ax_i.plot(t_sel_us, i_sel, label=f"Detected (Pad ±{PULSE_PAD_US} ), Aligned (Start {PULSE_START_US:.0f} µs)", **dict_STYLE_I_sel)

        # dashed black lines at window boundaries (robust for Series/ndarray)
        _t = np.asarray(t_sel_us)
        start_us, end_us = _t[0], _t[-1]
        for ax in (ax_v, ax_i):
            ax.axvline(start_us, color="k", linestyle="--", linewidth=1)
            ax.axvline(end_us,   color="k", linestyle="--", linewidth=1)

        # --- Dynamic x-range: window (already includes ±PULSE_PAD_US) plus extra margin
        xmin = float(start_us) - float(XRANGE_EXTRA_US)
        xmax = float(end_us)   + float(XRANGE_EXTRA_US)
        ax_i.set_xlim(xmin, xmax)   # sharex=True -> applies to ax_v too

        # --- X ticks: major every 100 µs, minor in between
        ax_i.xaxis.set_major_locator(MultipleLocator(X_MAJOR_TICK_US))
        ax_i.xaxis.set_minor_locator(MultipleLocator(X_MAJOR_TICK_US / max(1, X_MINOR_PER_MAJOR)))
        
        # Make majors/minors visually distinct
        ax_i.tick_params(axis="x", which="major", length=6, width=1)
        ax_i.tick_params(axis="x", which="minor", length=3, width=0.8)
        
        # (Optional) ensure plain numbers (no scientific notation)
        ax_i.ticklabel_format(axis="x", style="plain")
        
        # With sharex=True, ax_v inherits locators; keep only bottom labels visible
        for lbl in ax_v.get_xticklabels():
            lbl.set_visible(False)
        
        # sharex=True → ax_v inherits locators; keep only bottom labels visible
        ax_v.tick_params(axis="x", which="both", labelbottom=False)

        # Sub-Plot Titles (inside top-right of axes)
        ax_v.text(
            0.98, 0.95, "Voltage vs Time",
            transform=ax_v.transAxes,  # position in axes fraction (0–1)
            ha="right", va="top",
            fontsize=12, fontweight="bold"
        )

        ax_i.text(
            0.98, 0.95, "Current vs Time",
            transform=ax_i.transAxes,
            ha="right", va="top",
            fontsize=12, fontweight="bold"
        )
 
        # Legend Labels 
        label_axes(ax_v, ax_i)              # Calls heper function
        
        # Legend Title (after all lines are added)
        ax_v.legend(title="Before & After", loc="upper left", frameon=False)
        ax_i.legend(title="Before & After", loc="upper left", frameon=False)

        # ===== STAGE PLOTS per-file =====        
        # Stage this figure for saving later
        plot_file_name_base = f"PyPD_Plot_{file_name}_AutoOffset"
        plot_specs.append({
            "fig": fig,
            "ax_v": ax_v,
            "ax_i": ax_i,
            "plot_file_name_base": plot_file_name_base
        })

    # =========================
    # 🟠 📣 EXPORT (single call)
    # =========================
    pulse_params_snapshot = {
        "MIN_USEFUL_WIDTH_US": MIN_USEFUL_WIDTH_US,
        "NOISE_MULT": NOISE_MULT,
        "FALLBACK_THRESH_A": FALLBACK_THRESH_A,
        "FRAC_OF_PEAK": FRAC_OF_PEAK,
        "MIN_LOBE_WIDTH_US": MIN_LOBE_WIDTH_US,
        "AMP_SYMMETRY_BIAS": AMP_SYMMETRY_BIAS,
        "MAX_HOLE_WITHIN_US": MAX_HOLE_WITHIN_US,
        "MAX_INTERPHASE_GAP_US": MAX_INTERPHASE_GAP_US,
        "PULSE_PAD_US": PULSE_PAD_US,
        "PULSE_START_US": PULSE_START_US,
        "PULSE_SHIFT_US": PULSE_SHIFT_US,
        "XRANGE_EXTRA_US": XRANGE_EXTRA_US,
        "X_MAJOR_TICK_US": X_MAJOR_TICK_US,
        "X_MINOR_PER_MAJOR": X_MINOR_PER_MAJOR,
    }

    save_all_outputs(
        dict_CSV_RawTimeµs=dict_CSV_RawTimeµs,
        dict_CSV_Detected=dict_CSV_Detected,
        dict_CSV_AutoOffset=dict_CSV_AutoOffset,
        plot_specs=plot_specs,
        folder_path_parent=folder_path_parent,
        folder_path_output_main=folder_path_output_main,
        folder_path_output_dated=folder_path_output_dated,
        timestamp_str=timestamp_str,
        save_formats=SAVE_FORMATS,
        script_name=script_name,
        script_version=script_version,
        system_info=dict_SYSTEM_INFO,
        channel_map=dict_TEK_CHANNEL_MAP,
        axis_labels=dict_AXIS_LABELS,
        rc_subset_keys=RC_SUBSET_KEYS,
        pulse_params=pulse_params_snapshot,
        auto_open=True
    )

    # --------------------
    # Show Plots
    # --------------------
    # Show all figures (patched show will adjust y-label positions)
    plt.show()

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    load_tek_csv_files(file_path=None, skip_rows=15)
