"""
===============================================================================
NeuroEng Voltage Transient Processor - Pulse Detector (Stage 1)
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-08-29
Modified:   2025-10-30
Code:       VT_Processor_Py_Stage1_PulseDetector.py
Version:    v3.7.1


DESCRIPTION
---------------------------
This is **Stage 1** of the voltage transient processing pipeline.

This script prompts the user to select one or more raw Tektronix `.CSV` files.
For each file, it performs the following operations:
- **Loads and standardizes** the raw data, mapping columns to canonical
  Time (µs), Voltage (V), and Current (A).
- **Finds and Loads** the experiment metadata file (e.g., `Metadata.xlsx`)
  from the same parent directory, loading *only* the 18 specified base columns.
- **Saves** a `PyPD_Metadata_Loaded.CSV` of *only* this loaded data.
- **Calls Helper Module (`VT_Module_Py_DataIO`)** to process metadata:
  - Applies typing (e.g., Date as string).
  - Calculates 6 calculated fields (e.g., 'ElectrodeGSA(cm^2)' and
    all 'ChargePerPhase' columns).
  - **Re-orders** the final DataFrame to a specific 24-column format.
- **Saves** a `PyPD_Metadata_Calculated.CSV` of this processed, calculated,
  and re-ordered metadata (this is the primary input for Stage 2).
- **Auto-detects** the main biphasic pulse on the current channel.
- **Calculates** a baseline Y-offset (voltage and current) from the
  pre-pulse region and applies to create a *corrected* dataset.
- **Trims** the *corrected* data to this pulse window, adding defined padding
  (e.g., ±200 µs).
- **Time-offsets** the *trimmed, corrected* data so all pulses align to a
  canonical start time.
- **Saves** a diagnostic plot (`.png`) for each file, showing the *full
  corrected trace* overlaid with the *detected, corrected, and time-offset*
  pulse window.

OUTPUTS
---------------------------
- **`PyPD_Dictionary_AllDataFiles_wAutoOffset.pkl`**: The main output. A Python
  pickle file containing a dictionary where each key (e.g., "TEK00001")
  holds a DataFrame of the **baseline-corrected (Y-offset) and
  time-aligned (X-offset)** pulse data. This file is the
  input for Stage 2.
- **`PyPD_Metadata_Loaded.CSV`**: A snapshot of *only* the 18 loaded
  metadata columns, with no calculations.
- **`PyPD_Metadata_Calculated.CSV`**: A snapshot of the processed and
  re-ordered metadata (18 loaded + 6 calculated). **This is the input for Stage 2.**
- **`PyPD_Summary.json` / `.txt`**: JSON and Text files detailing all parameters,
  files, and settings used for the run.
- **Subfolders** containing the individual plots (`/PyPD_PNG_Files/`) and
  intermediate data (`/PyPD_CSV_Files/` and `/PyPD_ExtraInfo_Files/`):
  - `/PyPD_CSV_Files/`: Contains individual CSVs of the final
    baseline-corrected and time-offset data (matches the main .pkl).
  - `/PyPD_ExtraInfo_Files/`: Contains intermediate data:
    - `..._RawTimeµs.CSV`: The *original* standardized data,
      with no baseline or time correction.
    - `..._Detected.CSV`: The data *only* baseline-corrected
      (Y-offset), trimmed to the pulse but *not* time-offset.
      
===============================================================================
"""

#==============================================================================
# 🧠 AUTO-DETECT SCRIPT NAME & VERSION (from file name)
#==============================================================================
print("_"*40 + "")
import os
from pathlib import Path
from VT_Module_Py_System import get_script_info
script_name, script_version = get_script_info(__file__)
print(f"🟠 🧾 Running {script_name} ({script_version})")

#==============================================================================
# 📦 IMPORT 
#==============================================================================
print("🟠 📦 Importing Packages, Helper Modules...") 

#--------------------
# 📦 IMPORT REQUIRED PACKAGES
#-------------------- 
# --- Matplotlib backend configuration: prefer Spyder/Jupyter inline ---
import matplotlib
if os.environ.get("CI") or os.environ.get("HEADLESS") == "1":
    matplotlib.use("Agg")  # no windows in CI
else:
    # Prefer inline if inside IPython (Spyder/Jupyter)
    try:
        from IPython import get_ipython
        if get_ipython():
            matplotlib.use("module://matplotlib_inline.backend_inline")
        else:
            matplotlib.use("MacOSX")  # or "Qt5Agg"
    except Exception:
        matplotlib.use("Qt5Agg")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams   

#--------------------
# 🛠️ IMPORT HELPER MODULES (from .py)
#--------------------

# --- Settings & Constants ---
from VT_Module_Py_GlobalSettings import (
    DICT_SYSTEM_INFO, 
    DICT_TEK_CHANNEL_MAP,
    DICT_AXIS_LABELS, 
    CANON_TIME, CANON_VOLT, CANON_CURR, 
    # rcParams theme applied automatically on import
)

# --- System Interactions ---
from VT_Module_Py_System import (
    get_script_info,          # Moved from Utilities
    select_csv_files_gui,     # Moved from FileIO
    create_output_structure,  # Moved from Paths
    STAGE_PULSE_DETECTOR,     # Moved from Paths
    print_output_tree,        # Moved from TreeView
    print_latest_output_tree, # Moved from TreeView
)

# --- Data Input/Output ---
from VT_Module_Py_DataIO import (
    load_metadata_excel,
    process_metadata_snapshot,
    save_pipeline_artifacts,
    format_metadata_numeric
)

# --- Data Processing Algorithms ---
from VT_Module_Py_Processing import (
    standardize_dataframe_to_µs, # Moved from GlobalSettings
    find_biphasic_window,        # Moved from PulseAlgorithm
    # Import constants needed for the JSON summary snapshot & processing
    NOISE_MULT, 
    MIN_CURR_THRESHOLD_AMPS, 
    FRAC_OF_PEAK, 
    MIN_PHASE_WIDTH_µs, 
    MIN_PHASE_HOLE_WIDTH_µs,
    MAX_INTERPHASE_WIDTH_µs, 
    PULSE_AMPL_SYMMETRY_BIAS, 
    PULSE_START_µs, 
    PULSE_PAD_µs, 
    PULSE_OFFSET_µs, 
    XRANGE_EXTRA_µs, 
    X_MAJOR_TICK_µs, 
    X_MINOR_PER_MAJOR,
)

# --- Plotting Helpers ---
from VT_Module_Py_PlotHelpers import (
    place_legend_left, 
    # AUTO_YLABEL_ADJUST and patched_show are applied automatically on import
)

# --- User Input ---
# (No direct user input functions used in Stage 1)

# =============================================================================
# 🛠️ ADDITIONAL CONFIGURATION
# =============================================================================
print("🟠 🛠️ Applying Additional Configuration...")

#--------------------
# 🎨 PLOT SETTINGS
#--------------------

LIST_RC_SUBSET_KEYS = [
    "font.family",
    "font.size",
    "axes.labelsize",
    "axes.titlepad",
    "axes.titlesize",
    "axes.linewidth",
    "axes.facecolor",
    "axes.grid",
    "grid.color",
    "grid.linestyle",
    "grid.linewidth",
    "xtick.labelsize",
    "ytick.labelsize",
    "lines.linewidth",
    "figure.figsize",
    "savefig.dpi",
    "legend.fontsize",
    "legend.loc"
]

# Plot Style Dictionary (v3)
DICT_STYLE = {
    "voltage": {
        "raw": dict(color="#E6B800", linewidth=1.2, alpha=0.7, zorder=0, linestyle="-"),
        "sel": dict(color="Black", linewidth=1.5, alpha=1.0, zorder=3, linestyle="--"),
    },
    "current": {
        "raw": dict(color="#1E90FF", linewidth=1.2, alpha=0.7, zorder=0, linestyle="-"),
        "sel": dict(color="black", linewidth=1.5, alpha=1.0, zorder=3, linestyle="--"),
    },
}

SHOW_WINDOW_LINES = False   # True to show, False to hide

#--------------------
# 🌳 TREE VIEW CONFIG
#--------------------
SHOW_TREE  = True           # False to disable
TREE_MODE  = "current"       # "current", "latest", or "both"

#--------------------
# Global Data Dictionaries
#--------------------
DICT_CSV_RawTimeµs = {}      # original unmodified TEK CSV
DICT_CSV_Detected = {}  # detected biphasic window (not offset)
DICT_CSV_AutoOffset = {}   # detected window with time offset applied

#--------------------
# EXPORT SETTINGS
#--------------------
SAVE_FORMATS = ("png",)   # add more if you like, e.g., "eps"

#==============================================================================
# 🛠️ [HELPER FUNCTION] 
#==============================================================================

#====================
# 🔄 CSV Loader + 📊 Plotter
#====================

def main(file_path=None, skip_rows=15):
    """Runs the full Stage 1 Pulse Detector pipeline.
    ...
    """
    
    #===========================================
    # SELECT FILES, ORGANIZE FOLDERS
    #===========================================
    
    #---------------------------
    # 📁 FILE PICKER: Choose CSVs if not provided
    #---------------------------
    # 📣 [Call Function]  
    if not file_path:
        file_paths_list = select_csv_files_gui()
    else:
        file_paths_list = [Path(p) for p in file_path] # Convert all to Path
        print("   ✅ Using provided file path(s).")

    #---------------------------
    # 📁 IMPORT FOLDER
    #---------------------------
    # 📣 [Call Function]  
    folder_path_parent = file_paths_list[0].parent
    folder_name_parent = folder_path_parent.name

    #---------------------------
    # 📤 📂 CREATE EXPORT FOLDERS
    #---------------------------
    stage_name, stage_id = STAGE_PULSE_DETECTOR
    paths = create_output_structure(folder_path_parent, stage_name, stage_id)

    print(f"\n✅ Created Output Folders:"
          f"\n↳📂 {paths.output.name}"
          f"\n   ↳📂 {paths.stage.name}"
          f"\n      ↳📂 {paths.processed.name}"
          f"\n         ↳📂 {paths.csv.name}"
          f"\n         ↳📂 {paths.png.name}"
        + (f"\n         ↳📂 {paths.extra.name}" if paths.extra else "")
        + (f"\n         ↳📂 {paths.svg.name}" if paths.svg else "")
          )
          
    # =========================================================================
    # 📚 LOAD METADATA + PROCESS + SAVE SNAPSHOTS
    # =========================================================================
    
    # This list now contains *only* the 18 columns to *load*.
    META_COLS = [   
        # Base 18 columns to load
        "FileID", 
        "Date", 
        "WaferID", 
        "DeviceID", 
        "ElectrodeID", 
        "ElectrodeGeometry", 
        "ElectrodeDiameter(cm)", 
        "ElectrodeMaterial", 
        "Electrolyte", 
        "TestInfo1", 
        "TestInfo2",
        "WaveformID", 
        "Current(µA)", 
        "PhaseWidth(µs)", 
        "InterphaseDelay(µs)",
        "Frequency(Hz)", 
        "OtherInfo", 
        "TotalPulses",
    ]

    # ---- 1) Load Metadata file from the parent folder ----
    # This will now *only* load the columns specified in META_COLS
    DF_Metadata_Loaded = load_metadata_excel(paths.parent, META_COLS)
    
    # ---- 2a) Save the "LOADED" data only----
    meta_csv_loaded_name = "PyPD_Metadata_Loaded.CSV"
    meta_csv_loaded_path = paths.stage / meta_csv_loaded_name
    # Loaded snapshot
    DF_Metadata_Loaded = load_metadata_excel(paths.parent, META_COLS)
    DF_Metadata_Loaded = format_metadata_numeric(DF_Metadata_Loaded)
    DF_Metadata_Loaded.to_csv(meta_csv_loaded_path, index=False)
    print(f"✅ 💾 Saved Metadata Loaded: {meta_csv_loaded_name}") 

    # ---- 2b) ⭐️ Process Metadata (types, calculations, re-order) ----
    # 📣 [Call Helper Fn] from VT_Module_Py_DataIO
    # We pass a *copy* to avoid modifying the original loaded data
    DF_Metadata_Calculated = process_metadata_snapshot(DF_Metadata_Loaded.copy())
    
    # ---- 2c) Save the "CALCULATED" (for Stage 2) ----
    meta_csv_calculated_name = "PyPD_Metadata_Calculated.CSV"
    meta_csv_calculated_path = paths.stage / meta_csv_calculated_name
    # Calculated snapshot
    DF_Metadata_Calculated = process_metadata_snapshot(DF_Metadata_Loaded.copy())
    DF_Metadata_Calculated = format_metadata_numeric(DF_Metadata_Calculated)
    DF_Metadata_Calculated.to_csv(meta_csv_calculated_path, index=False)
    print(f"✅ 💾 Saved CALCULATED Metadata: {meta_csv_calculated_name}")
    
    #===========================================
    # PROCESS EACH CSV FILE
    #===========================================
    print("\n🟠 Detecting Pulse...")
    
    plot_specs = [] # *** Initialize list *before* the loop ***
    
    for csv_path in file_paths_list: 
        file_name = csv_path.stem
        
        #--------------------
        # 1) Read CSV (header around row 16)
        #--------------------
        try:
            df = pd.read_csv(csv_path, skiprows=skip_rows, sep=",", engine="c")
        except Exception:
            df = pd.read_csv(csv_path, skiprows=skip_rows, sep=None, engine="python")

        #--------------------
        # 2) Standardize Data
        #--------------------
        try:
            df_std = standardize_dataframe_to_µs(df, channel_map=DICT_TEK_CHANNEL_MAP)
        except Exception as e:
            print(f"❌ {file_name}: Failed to standardize. {e}. Skipping.")
            continue

        #--------------------
        # 3) Get data as numpy arrays from standardized DF
        #--------------------
        t_µs  = df_std[CANON_TIME].to_numpy() # Already in µs
        v_raw = df_std[CANON_VOLT].to_numpy()
        i_raw = df_std[CANON_CURR].to_numpy()

        #--------------------
        # 4) Auto-detect biphasic window on CURRENT
        #--------------------
        found_pulse = True
        found = find_biphasic_window(t_µs, i_raw) # Pass µs time
        if found is None:
            print(f"❗️{file_name}: No biphasic pulse found → plotting RAW only.")
            found_pulse = False
        else:
            i0, i1, thr = found
            if i1 <= i0:
                print(f"❗{file_name}: Invalid window indices ({i0}, {i1}) → plotting RAW only.")
                found_pulse = False
        
        #--------------------
        # 5) Build data depending on whether we found a pulse
        #--------------------
        
        # 5a) Offset-Y (Correct Voltage and Current Drift)
        v_offset = 0.0
        i_offset = 0.0
        if found_pulse and i0 > 0:
            # Calculate offset from the raw baseline (everything before index i0)
            v_offset = v_raw[0:i0].mean()
            i_offset = i_raw[0:i0].mean()
            print(f"   ℹ️ {file_name}: Applying V-Offset: {v_offset:.6f} V, C-Offset: {i_offset:.6f} A")
        elif not found_pulse:
            print(f"   ℹ️ {file_name}: No pulse found. Skipping Y-Offset.")
        else: # found_pulse is True but i0 is 0 (pulse starts at beginning)
            print(f"   ℹ️ {file_name}: No baseline (i0=0). Skipping Y-Offset.")

        # Create new corrected arrays
        v_cor = v_raw - v_offset
        i_cor = i_raw - i_offset
        
        # 5b) Offset-X (Correct Time Drift)
        if found_pulse:
            # Trim + Offset (so selected starts at user defined "PULSE_START_µs" µs)
            df_sel_time_ref = df_std.iloc[i0:i1+1].copy() 
            
            # 1. Get start time directly (in µs)
            t_start_sel_µs = df_sel_time_ref[CANON_TIME].iloc[0]
            
            # 2. Calculate the PULSE_OFFSET (in µs) e.g., -200 µs
            t_offset_µs = PULSE_OFFSET_µs - t_start_sel_µs
        
            # 3. Apply the µs Offset to time
            t_raw_µs = df_std[CANON_TIME] + t_offset_µs
            t_sel_µs = df_std[CANON_TIME].iloc[i0:i1+1] + t_offset_µs # Apply offset to trimmed time
            
            # 4. Get the selected window from the *corrected* data
            v_sel = v_cor[i0:i1+1] 
            i_sel = i_cor[i0:i1+1]
        
        else:
            # RAW only: no offset alignment 
            t_raw_µs = df_std[CANON_TIME]
            t_sel_µs = None  # sentinel
            v_sel = None     # sentinel
            i_sel = None     # sentinel
            
        #--------------------
        # 6) Build DataFrames for output dictionaries
        #--------------------
        
        # Raw (full TEK trace, no corrections)
        DF_OUT_RawTimeµs = df_std.copy() 
        
        if found_pulse:
            # Detected window (Y-corrected, but original time)
            DF_OUT_Detected = pd.DataFrame({
                CANON_TIME: t_µs[i0:i1+1],
                CANON_VOLT: v_cor[i0:i1+1],
                CANON_CURR: i_cor[i0:i1+1],
            })
            
            # AutoOffset window (Y-corrected AND X-offset)
            DF_OUT_AutoOffset = pd.DataFrame({
                CANON_TIME: t_sel_µs.to_numpy(),
                CANON_VOLT: v_sel,
                CANON_CURR: i_sel,
            })
        else:
            # Empty placeholder (no pulse found)
            cols = [CANON_TIME, CANON_VOLT, CANON_CURR]
            DF_OUT_Detected = pd.DataFrame(columns=cols)
            DF_OUT_AutoOffset   = pd.DataFrame(columns=cols)
        
        # Store all dictionaries
        DICT_CSV_RawTimeµs[file_name] = DF_OUT_RawTimeµs
        DICT_CSV_Detected[file_name]   = DF_OUT_Detected
        DICT_CSV_AutoOffset[file_name] = DF_OUT_AutoOffset

        #--------------------
        # 7) Plot Panel (Voltage top, Current bottom)
        #-------------------- 
        fig, (ax_v, ax_i) = plt.subplots(
            2, 
            1, 
            sharex=True,
            figsize=(rcParams["figure.figsize"][0], rcParams["figure.figsize"][1] * 1.2)
        )
        
        # Figure Title
        fig.suptitle(f"{folder_name_parent}_$\\bf{{{file_name}}}$")
        
        # Always plot RAW Plot (x-axis, y-axis, label, style)
        ax_v.plot(t_raw_µs, v_cor, **DICT_STYLE["voltage"]["raw"], label="Full Corrected Trace")
        ax_i.plot(t_raw_µs, i_cor, **DICT_STYLE["current"]["raw"], label="Full Corrected Trace")

        # Plot Detected+Offset only if we have it
        if found_pulse:
            ax_v.plot(
                t_sel_µs, v_sel,
                **DICT_STYLE["voltage"]["sel"],
                label=f"Detected (Pad ±{PULSE_PAD_µs} µs), Offset (Onset {PULSE_START_µs:.0f} µs)"
            )
            ax_i.plot(
                t_sel_µs, i_sel,
                **DICT_STYLE["current"]["sel"],
                label=f"Detected (Pad ±{PULSE_PAD_µs} µs), Offset (Onset {PULSE_START_µs:.0f} µs)"
            )

        # Dynamic X-range (includes ±PULSE_PAD_µs & extra margin )
        if found_pulse:
            start_µs, end_µs = float(t_sel_µs.iloc[0]), float(t_sel_µs.iloc[-1])
            xmin = start_µs - float(XRANGE_EXTRA_µs)
            xmax = end_µs   + float(XRANGE_EXTRA_µs)
        else:
            # Use full RAW span when no detection
            xmin = float(t_raw_µs.iloc[0]); xmax = float(t_raw_µs.iloc[-1])
        ax_i.set_xlim(xmin, xmax)   # sharex=True -> applies to ax_v too

        # Optional window lines only when detected and globally enabled
        if found_pulse and SHOW_WINDOW_LINES:
            for ax in (ax_v, ax_i):
                ax.axvline(start_µs, color="k", linestyle="--", linewidth=1)
                ax.axvline(end_µs,   color="k", linestyle="--", linewidth=1)

        # (Optional) ensure plain numbers (no scientific notation)
        ax_i.ticklabel_format(axis="x", style="plain")
        
        # With sharex=True, ax_v inherits locators; keep only bottom labels visible
        for lbl in ax_v.get_xticklabels():
            lbl.set_visible(False)
        
        # sharex=True → ax_v inherits locators; keep only bottom labels visible
        ax_v.tick_params(axis="x", which="both", labelbottom=False)
         
        # Axis labels (plain, no manual transform)
        ax_v.set_ylabel(DICT_AXIS_LABELS["VOLTAGE"])
        ax_i.set_ylabel(DICT_AXIS_LABELS["CURRENT"])
        ax_i.set_xlabel(DICT_AXIS_LABELS["TIME"])
        
        # Align the two y-labels to the same x-position
        fig.align_ylabels([ax_v, ax_i])

        # Legend (after plotting)
        if found_pulse:
            place_legend_left(ax_v); place_legend_left(ax_i)
        else:
            ax_v.legend(["Full Corrected Trace (no pulse detected)"], loc="upper left", frameon=False)
            ax_i.legend(["Full Corrected Trace (no pulse detected)"], loc="upper left", frameon=False)

        # ===== STAGE PLOTS per-file =====        
        # Stage this figure for saving later
        plot_file_name_base = f"PyPD_{file_name}_AutoOffset"
        plot_specs.append({
            "fig": fig,
            "ax_v": ax_v,
            "ax_i": ax_i,
            "plot_file_name_base": plot_file_name_base
        })
        
        print(f"✅ Processed: {file_name}") # *** ADDED: Confirmation per file ***

    # =========================
    # 🟠 📣 EXPORT (single call)
    # =========================

    # 1. Define Figure dictionary
    figures_to_save = {
        spec["plot_file_name_base"]: spec["fig"] 
        for spec in plot_specs
    }

    # 2. Define DataFrame dictionaries (by destination folder)
    # These go into the MAIN CSV folder
    dataframes_csv = {
        f"PyPD_{fname}_AutoOffset.CSV": df 
        for fname, df in DICT_CSV_AutoOffset.items()
    }
    # These go into the EXTRA folder
    dataframes_extra = {}
    for fname, df in DICT_CSV_RawTimeµs.items():
        dataframes_extra[f"PyPD_{fname}_RawTimeµs.CSV"] = df
    for fname, df in DICT_CSV_Detected.items():
        dataframes_extra[f"PyPD_{fname}_Detected.CSV"] = df

    # 3. Define Pickle dictionaries (by destination folder)
    # This goes into the STAGE folder
    pickles_stage = {
        "PyPD_Dictionary_AllDataFiles_wAutoOffset.pkl": DICT_CSV_AutoOffset
    }
    # These go into the EXTRA folder
    pickles_extra = {
        "PyPD_Dictionary_AllDataFiles_wRawTimeµs.pkl": DICT_CSV_RawTimeµs,
        "PyPD_Dictionary_AllDataFiles_wDetected.pkl": DICT_CSV_Detected
    }

    # 4. Assemble the PyPD_Summary dictionary
    # 4a. Get rcparams
    rcparams_subset = {k: rcParams[k] for k in LIST_RC_SUBSET_KEYS if k in rcParams}
    
    # 4b. Get offset summary
    offset_summary_µs = {}
    for fname, DF_OFFSET in DICT_CSV_AutoOffset.items():
        DF_DETECTED = DICT_CSV_Detected.get(fname)
    
        if DF_OFFSET is None or DF_DETECTED is None or DF_OFFSET.empty or DF_DETECTED.empty:
            offset_summary_µs[fname] = {
                "status": "raw_only", "detected_start_µs": None,
                "offset_start_µs": None, "delta_µs": None,
            }
            continue
        try:
            t_off_start = float(DF_OFFSET[CANON_TIME].iloc[0])
            t_det_start = float(DF_DETECTED[CANON_TIME].iloc[0])
            offset_summary_µs[fname] = {
                "status": "detected",
                "detected_start_µs": t_det_start,
                "offset_start_µs":   t_off_start,
                "delta_µs":          t_off_start - t_det_start,
            }
        except Exception:
            offset_summary_µs[fname] = {
                "status": "error", "detected_start_µs": None,
                "offset_start_µs": None, "delta_µs": None,
            }
            
    # 4c. Get artifact file lists
    saved_plots = [f"{name}.{ext}" for name in figures_to_save.keys() for ext in SAVE_FORMATS]
    saved_csvs = list(dataframes_csv.keys()) + list(dataframes_extra.keys())
    saved_pickles = list(pickles_stage.keys()) + list(pickles_extra.keys())
    
    # 4d. Build the final summary dictionary
    DICT_Pulse_Detection_Params = {
        "NOISE_MULT": NOISE_MULT,
        "MIN_CURR_THRESHOLD_AMPS": MIN_CURR_THRESHOLD_AMPS,
        "FRAC_OF_PEAK": FRAC_OF_PEAK,
        "MIN_PHASE_WIDTH_µs": MIN_PHASE_WIDTH_µs,
        "PULSE_AMPL_SYMMETRY_BIAS": PULSE_AMPL_SYMMETRY_BIAS,
        "MIN_PHASE_HOLE_WIDTH_µs": MIN_PHASE_HOLE_WIDTH_µs,
        "MAX_INTERPHASE_WIDTH_µs": MAX_INTERPHASE_WIDTH_µs,
        "PULSE_PAD_µs": PULSE_PAD_µs,
        "PULSE_START_µs": PULSE_START_µs,
        "PULSE_OFFSET_µs": PULSE_OFFSET_µs,
        "XRANGE_EXTRA_µs": XRANGE_EXTRA_µs,
        "X_MAJOR_TICK_µs": X_MAJOR_TICK_µs,
        "X_MINOR_PER_MAJOR": X_MINOR_PER_MAJOR,
    }
    
    PyPD_Summary = {
        "timestamp": paths.timestamp,
        "script": {"name": script_name, "version": script_version},
        "system": DICT_SYSTEM_INFO,
        "input_folder": str(paths.parent),  
        "output_folder": str(paths.stage), 
        "n_files_processed": len(DICT_CSV_RawTimeµs),
        "file_keys": list(DICT_CSV_RawTimeµs.keys()),
        "channel_map": DICT_TEK_CHANNEL_MAP,
        "axis_labels": DICT_AXIS_LABELS,
        "rcparams_subset": rcparams_subset,
        "pulse_params": DICT_Pulse_Detection_Params,
        "offset_summary_µs": offset_summary_µs,
        "artifacts": {
            # ⭐️ MODIFIED v3.7.1: Use new, clearer keys
            "metadata_loaded": meta_csv_loaded_name, 
            "metadata_calculated": meta_csv_calculated_name,
            "plots": saved_plots,
            "csvs": saved_csvs,
            "pickles": saved_pickles,
        }
    }

    # 5. Make the single call to the exporter
    save_pipeline_artifacts(
        paths=paths,
        summary_data=PyPD_Summary,
        summary_base_name="PyPD_Summary",
        figures_to_save=figures_to_save,
        dataframes_csv=dataframes_csv,
        dataframes_extra=dataframes_extra,
        pickles_stage=pickles_stage,
        pickles_extra=pickles_extra,
        save_formats=SAVE_FORMATS
    )
    
    #===================== 
    # 🌳 OUTPUT FOLDER TREE (optional)
    #=====================\
    if SHOW_TREE:
        try:
            if TREE_MODE in ("current", "both"):
                print("\n🌳 Folder Tree:")
                print_output_tree(paths.stage)
            if TREE_MODE in ("latest", "both"):
                print_latest_output_tree(paths.output, stage_name=stage_name)
        except Exception as e:
            print(f"🚫 Folder tree preview failed: {e}")
            
    #=====================\
    # Show Plots
    #=====================\
    # Show all figures (patched show will adjust y-label positions)
    plt.show()

# =============================================================================
# 🚀 Run
# =============================================================================
if __name__ == "__main__":
    main(file_path=None, skip_rows=15)

#=============================================================================
# Code Complete
#=============================================================================
print("_"*40 + "")
print("🎉 Completed: Py_Stage1_PulseDetector")
print("👉 Now run:   Py_Stage2_VoltageSelector")
print("_"*40 + "")

