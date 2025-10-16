# -*- coding: utf-8 -*-
"""
===============================================================================
NeuroEng Voltage Transient Processor - Data Overlay
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-07-07
Modified:   2025-10-15
Version:    v1

REQUIREMENTS
---------------------------
- Works in Spyder, VS Code, or terminal — no coding required
- Python packages: pandas, matplotlib, tkinter (all standard)

DESCRIPTION
---------------------------
This script provides a user-friendly GUI to process voltage transient (VT) data 
collected from a Tektronix oscilloscope (TEK0000x.CSV format).

Features:
- Select and overlay multiple TEK0000x.CSV files using a GUI file picker.
- Automatically rename columns and convert time from seconds to milliseconds.
- Plot Voltage and Current traces on shared axes.
- Optionally shift time (offset) to align traces from 0 ms.
- Customize x-axis (time) limits directly in the terminal.
- Save all plots as PNG, SVG, and EPS with today’s date.
- Export raw and offset CSVs, plus a time-offset summary log.

WORKFLOW
---------------------------
- Input:
    - CSVs must have headers: 'TIME', 'CH1', 'CH2'
    - Data begins on row 16 (skiprows=15)
    - 'TIME' → converted to 'Time (ms)'
    - 'CH1' → 'Voltage (V)', 'CH2' → 'Current (A)'

- Output (in 'VT_Py_Outputs' folder):
    - "Voltage vs Time" and "Current vs Time plots (initial and offset)
    - Date-stamped CSVs for raw and offset traces
    - A summary .txt file showing applied time shifts



EMOJI LEGEND
---------------------------
    ❗   ACTION REQUIRED         - User must make a choice or input something
    ✅   SUCCESS / COMPLETED     - A step was successfully completed
    🟠   IN PROGRESS             - Something is currently running or processing
    🚫   ERROR / EXIT            - Something failed or script is exiting
    ❌   CANCELED / SKIPPED      - User skipped or canceled a step
    📁   FILE PICKER             - Selecting files or folders
    📊   PLOTTING                - Displaying or generating plots
    💾   FILE SAVE               - Files (plots, CSVs) are being saved
    📝   LOG CREATED             - Summary or text logs saved
    ✏️    USER INPUT              - User must enter a value (e.g. offset, axis limit)
    ⏱️    TIME SCALE SET          - Time range input confirmed
    🖋️   ANNOTATION              - Plot labels or arrows being added
    🛠️   CONFIGURATION           - Customize parameters (figure size, fonts, DPI, legend)
    🔁   RESTART                 - Prompt to rerun the script
    🌟🎉 COMPLETE                - Final success message at the end

"""
# =============================================================================
# 🧭 SCRIPT ROADMAP (JUMP TO SECTION)
# =============================================================================
# 1. 🛠️ Global Config & Defaults
# 2. 📁 File Selection (GUI)
# 3. 🔄 Load CSVs + Initial Plot Preview
# 4. 🧠 User Inputs: Offset, Axis, Titles, Labels
# 5. 📊 Create Final Plots
# 6. 📤 Export: Plots, CSVs, Pickles, Summary
# 7. 📂 Auto-Open Output Folder & Summary
# 8. 🌟 Completion Message

# =============================================================================
# 📦 IMPORT REQUIRED PACKAGES
# =============================================================================
print("🟠 📦 Importing Packages...")

# --- Core Python ---
import os                           # File and directory operations
import sys                          # System exit and interpreter control
import time                         # Timing delays and timestamps
from datetime import datetime       # For date-stamping saved files

import platform
import getpass
import json
import pickle

# --- Data Handling & Plotting ---
import pandas as pd                 # For CSV reading and data manipulation
import matplotlib
import matplotlib.pyplot as plt     # For plotting
from matplotlib import rcParams     # For global plot styling

# --- GUI (Tkinter) ---
from tkinter import Tk, filedialog  # Basic file dialogs and root window
    
# =============================================================================
# 🛠️ GLOBAL CONFIGURATION
# =============================================================================
print("🟠 🛠️ Global Configuration...")

# --------------------
# Script PyDO_Summary
# --------------------

script_name = "VT_Plotter_Batch_Overlay.py"
script_version = "2025-08-05"
df_system_info = {
    "system": platform.system(),
    "release": platform.release(),
    "python_version": platform.python_version(),
    "user": getpass.getuser()
}

# --------------------
# 🛠️ 📦 Default Global Settings
# --------------------

# File input
skip_rows_default = 15
file_picker_title = "Select CSV files"

# Column names
xcol_t = "Time (ms)"
ycol_v = "Voltage (V)"
ycol_i = "Current (A)"

# Plot titles
plot_title_v_default = "Voltage vs Time"
plot_title_i_default = "Current vs Time"

# Axis limits
xaxis_min_default = -0.1
xaxis_max_default = 0.7

# Legend
legend_title_default = None

# File naming
custom_suffix_default = ""

# Save formats
save_eps_default = False
save_formats = ["png", "svg"] + (["eps"] if save_eps_default else [])

# --------------------
# 🎨 Global Plot Style (matplotlib.rcParams)
# --------------------

rcParams.update({

    # --- 🖋️ Font & Layout ---
    "font.family": "Arial",              # Global font family   # e.g., "Arial", "Times New Roman", "Helvetica", "DejaVu Sans"
    "font.size": 10,                     # Base font size       # e.g., 8, 10, 12
    "font.weight": "normal",             # Global font weight   # "normal", "bold", "light", "medium", "heavy"
    "axes.labelsize": 12,                # Axis label size      # integer (e.g., 10, 12, 14)
    "axes.labelweight": "normal",        # Axis label weight    # "normal", "bold"
    "axes.titlesize": 14,                # Plot title size      # integer (e.g., 14, 16)
    "axes.titleweight": "normal",        # Plot title weight    # "normal", "bold"
    "axes.titlepad": 10,                 # Space between title and plot  # int/float (padding in points)

    # --- 📐 Axes Style ---
    "axes.linewidth": 1,                 # Border thickness     # float (e.g., 0.8, 1.0, 1.5)
    "axes.facecolor": "white",           # Plot background      # e.g., "white", "lightgray", "#f0f0f0"
    "axes.grid": False,                  # Enable gridlines     # True or False

    # --- 📊 Grid Style ---              # Applies if axes.grid = True
    "grid.color": "#dddddd",             # Grid line color      # e.g., "#cccccc", "lightgray"
    "grid.linestyle": "--",              # Grid line style      # "-", "--", ":", "-."
    "grid.linewidth": 0.5,               # Grid line thickness  # float (e.g., 0.3, 0.5, 1.0)

    # --- 📏 Tick Style ---
    "xtick.labelsize": 10,               # X tick font size     # int (e.g., 8–12)
    "ytick.labelsize": 10,               # Y tick font size     # int
    "xtick.major.width": 1,              # X tick thickness     # float
    "ytick.major.width": 1,              # Y tick thickness     # float
    "xtick.direction": "out",            # X tick direction     # "in", "out", "inout"
    "ytick.direction": "out",            # Y tick direction     # "in", "out", "inout"

    # --- 📈 Line Style ---
    "lines.linewidth": 1,                # Default line width   # float (e.g., 1.0, 2.0)

    # --- 🖼️ Figure Layout ---
    "figure.figsize": (7.2, 4),          # Default plot size    # tuple (width, height) in inches
    "figure.autolayout": True,           # Auto-layout on save  # True = avoids clipping axis labels

    # --- 💾 Save Style ---
    "savefig.dpi": 300,                  # Output resolution    # e.g., 100, 200, 300 (for high-res export)

    # --- 🗂️ Legend Style ---
    "legend.title_fontsize": 10,         # Legend title font    # int
    "legend.fontsize": 9,                # Legend text size     # int
    "legend.loc": "best",                # Legend placement     # "best", "upper right", "lower left", etc.
    "legend.frameon": False,             # Show legend box      # True or False
    "legend.edgecolor": "gray",          # Legend box color     # ignored if frame is off
    "legend.fancybox": True,             # Rounded box corners  # True or False
    "legend.borderpad": 0.4,             # Space around legend  # float

    # --- 🔠 Text Rendering ---
    "text.usetex": False,                # Use LaTeX            # True (pretty, slow) or False (faster)
})

# --------------------
# 🔁 USER INPUT HELPER FUNCTION
# --------------------

def get_user_input(prompt_text, default_value):
    """
    Prompt user for input. If user presses Enter, return default_value.
    """
    user_input = input(f"{prompt_text} [Press Enter for '{default_value}']: ").strip()
    return user_input if user_input else default_value
    
# =============================================================================
# 🔄 FUNCTION: LOAD CSV + 📊 INITIAL PLOTS
# =============================================================================

def load_tek_csv_files(file_path, skip_rows, xcol_t, ycol_v, ycol_i):
    
    # --------------------
    # 📁 GUI FILE SELECTION: Choose CSVs if not provided
    # --------------------
    if not file_path:
        print("🟠 🖥️ Opening File Picker to select TEK.CSV files...")
        print("\n❗ ACTION REQUIRED ❗")
        print("👉 📁 Select one or more TEK.CSV files to process")

        Tk().withdraw()
        file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

        if not file_path:
            print("\n🚫 No files selected. Exiting script.")
            print("_" * 80)
            sys.exit()

    # Base folder where TEK.CSVs came from
    folder_path_parent = os.path.dirname(file_path[0])
    folder_name_parent = os.path.basename(folder_path_parent)
    print(f"   ✅ Selected folder: → {folder_path_parent}")
    print("   ✅ Selected files:")
    for i, path in enumerate(file_path, start=1):
        print(f"      {i}. {os.path.basename(path)}")

    # --------------------
    # 🔄 LOAD CSV + 📊 INITIAL PLOTS
    # --------------------
    print("\n🟠 🔄 Loading selected files...")
    dict_df_raw = {}  # key = file_name, value = dataframe

    print("🟠 📊 Initializing preview figures...")
    fig_v_init, ax_v_init = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    fig_i_init, ax_i_init = plt.subplots(figsize=plt.rcParams["figure.figsize"])

    for csv_path in file_path:
        file_selected = os.path.basename(csv_path)
        file_name = os.path.splitext(file_selected)[0]

        try:
            df_raw = pd.read_csv(csv_path, skiprows=skip_rows)
            df_raw.rename(columns={
                'TIME': 'Time (s)',
                'CH1': ycol_v,
                'CH2': ycol_i
            }, inplace=True)

            df_raw[xcol_t] = df_raw['Time (s)'] * 1000
            dict_df_raw[file_name] = df_raw

            ax_v_init.plot(df_raw[xcol_t], df_raw[ycol_v], label=file_name)
            ax_i_init.plot(df_raw[xcol_t], df_raw[ycol_i], label=file_name)

        except Exception as e:
            print(f"🚫 Error reading {file_selected}: {e}")

    for ax, title, ylabel in zip(
        [ax_v_init, ax_i_init],
        [plot_title_v_default, plot_title_i_default],
        [ycol_v, ycol_i]
    ):
        ax.set_title(title)
        ax.set_xlabel(xcol_t)
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.show()

    if not matplotlib.is_interactive():
        input("🛑 🖼️ Plot shown. 👉 Press ENTER to continue...")

    return folder_path_parent, dict_df_raw, fig_v_init, ax_v_init, fig_i_init, ax_i_init

# 🟠 📣 [Call Function] - LOAD CSV + 📊 INITIAL PLOTS
folder_path_parent, dict_df_raw, fig_v_init, ax_v_init, fig_i_init, ax_i_init = load_tek_csv_files(
    file_path=None,
    skip_rows=skip_rows_default,
    xcol_t=xcol_t,
    ycol_v=ycol_v,
    ycol_i=ycol_i
)

# =============================================================================
# 👉 USER INPUT:
# =============================================================================

# --------------------
# 👉 ❓ CONTINUE OR RESTART?
# --------------------

while True:
    print("\n❗ ACTION REQUIRED ❗")
    user_choice = input("👉 ❓ Continue plotting or restart? (Y/N): ").strip().upper()
    if user_choice == 'Y':
        break
    elif user_choice == 'N':
        print("\n❌ User aborted. ")
        print("🟠 Closing plot windows...")
        print("🟠 Exiting script...")
        print("👉 Note: Plots and Variable Explorer must be cleared manually in IDEs.")
        plt.close('all')    # Close all open plot windows
        print("🔁 Script terminated. Please re-run code.")
        print("_" * 80)
        time.sleep(3)
        raise SystemExit
    else:
        print("🚫 Invalid input. Type Y/N.")
        
# 🟠 📣 [Call Function] 

# --------------------
# 👉 ❓ Offset Traces (Console-based)
# --------------------

def get_user_offset_values(dict_df_raw, xcol_t):
    dict_df_processed = {}
    user_offset_summary = {}
    user_offset_choice = input("\n👉 ❓ Apply time offset to any traces? (Y/N): ").strip().upper()
    
    if user_offset_choice == "Y":
        print("👉 🖊️ Input Offset Values (ms):")
        print("      ➕ Enter -0.6 → to shift from -0.6 ms to 0 ms → ")
        print("      ➕ Enter 0.0 → to leave it unchanged")
        for file_name, df_raw in dict_df_raw.items():
            while True:
                try:
                    user_offset_val = float(input(f"   ✏️ Offset for '{file_name}': "))
                    df_user_offset = df_raw.copy()
                    df_user_offset[xcol_t] -= user_offset_val
                    dict_df_processed[file_name + "_UserOffset"] = df_user_offset
                    user_offset_summary[file_name] = user_offset_val
                    break
                except ValueError:
                    print("   🚫 Invalid input. Please enter a number.")
        print("   ✅ Offset applied for all traces.") 
          
    elif user_offset_choice == "N":
        print("   ❌ Offset skipped for all traces.")
        for file_name, df_raw in dict_df_raw.items():
            dict_df_processed[file_name + "_Raw"] = df_raw.copy()
            user_offset_summary[file_name] = 0.0
    
    else:
        print("🚫 Invalid input. Type Y or N.")
        sys.exit()
    
    return dict_df_processed, user_offset_summary

# 🟠 📣 [Call Function] - 🧭 Time Offset
dict_df_processed, user_offset_summary = get_user_offset_values(dict_df_raw, xcol_t)

# --------------------
# 👉 🧭 Custom Time Scale
# --------------------

def get_time_limits_console(default_min=xaxis_min_default, default_max=xaxis_max_default):
    print("\n👉 🧭 Set Time Scale (ms): ")
    try:
        xmin = float(get_user_input("   ✏️ 🔽 Min (ms)", default_min))
        xmax = float(get_user_input("   ✏️ 🔼 Max (ms)", default_max))
    except ValueError:
        print("🚫 Invalid input. Using default values.")
        xmin, xmax = default_min, default_max
    print(f"   ✅ Time Scale (ms): {xmin} , {xmax}")
    return xmin, xmax

# 🟠 📣 [Call Function] - 🧭 Time Scale (x-axis)
xmin_t, xmax_t = get_time_limits_console()

# --------------------
# 👉 🏷️ Legend Labels
# --------------------

def get_custom_legends(dict_df_processed):
    legend_label = {}
    print("\n👉 🖊️ Input Legend Labels: ")
    print("      ➕ [Press Enter for 'TEK000xx']")
    for file_key in dict_df_processed:
        base = file_key.replace("_user_offset", "")
        label = input(f"   ✏️ Label for {base}: ").strip()
        if label:
            legend_label[file_key] = label
        else:
            legend_label[file_key] = base
    print("   ✅ Legend Labels applied to all traces")
    return legend_label

# 🟠 📣 [Call Function] - 🏷️ Legend Labels
legend_label = get_custom_legends(dict_df_processed)

# --------------------
# 👉 🏷️ Legend Title (Optional)
# --------------------

def get_legend_title(default_title):
    print("\n👉 🖊️ Input Legend Title:")
    legend_title = get_user_input("   ✏️ Legend Title: ", legend_title_default)
    if legend_title:
        print(f"   ✅ Legend Title: {legend_title}")
    else:
        print("   ❌ No Legend Title")
    return legend_title

# 🟠 📣 [Call Function] - 🏷️ Legend Title
legend_title = get_legend_title(legend_title_default)

# --------------------
# 👉 Plot Title
# --------------------

def get_plot_titles(default_v, default_i):
    print ("\n👉 🖊️ Input Plot Title:")
    plot_title_v = get_user_input("   ✏️ Title for V plot: ", plot_title_v_default)
    plot_title_i = get_user_input("   ✏️ Title for I plot: ", plot_title_i_default)
    print(f"   ✅ Plot Titles: {plot_title_v} , {plot_title_i}")
    return plot_title_v, plot_title_i

# 🟠 📣 [Call Function] -🖊️ Plot Titles
plot_title_v, plot_title_i = get_plot_titles(plot_title_v_default, plot_title_i_default)

# --------------------
# 👉 Plot File Name Suffix
# --------------------

# 🧠 Extract base experiment name (e.g., "20250616_IF0011_IIA42")
folder_name_parent = os.path.basename(folder_path_parent)
base_exp_name = folder_name_parent.split("_VT")[0] if "_VT" in folder_name_parent else folder_name_parent

# ✅ Show example output plot names
print("\n👉 🖊️ Add Suffix to Plot File Names?")
print("     ➕ Suffix: E01, VarCurrent, VarIPdelay, 1Billion")
print(f"     ➕ File Name:  {base_exp_name}_VTplots_V_<suffix>.png")

custom_suffix = get_user_input("   ✏️ Custom Suffix", custom_suffix_default).replace(" ", "-")
if custom_suffix:
    print(f"   ✅ Plot File Name Suffix: {custom_suffix}")
else:
    print("   ❌ No Suffix")

# =============================================================================
# 📊 CREATE FINAL PLOTS
# =============================================================================

def create_final_plots(
    dict_df_processed,
    legend_label,
    plot_title_v,
    plot_title_i,
    ycol_v,
    ycol_i,
    xcol_t,
    xmin_t,
    xmax_t,
    legend_title
):
    print("\n🟠 Generating Final Plots...")
    
    # Initialize plot figures
    fig_v_final, ax_v_final = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    fig_i_final, ax_i_final = plt.subplots(figsize=plt.rcParams["figure.figsize"])

    # Plot each offset trace
    for file_name, df_user_offset in dict_df_processed.items():
        label = legend_label.get(file_name, file_name)
        ax_v_final.plot(df_user_offset[xcol_t], df_user_offset[ycol_v], label=label)
        ax_i_final.plot(df_user_offset[xcol_t], df_user_offset[ycol_i], label=label)
    
    # Style both plots
    for ax, title, ylabel in zip(
        [ax_v_final, ax_i_final],
        [plot_title_v, plot_title_i],
        [ycol_v, ycol_i]
    ):
        ax.set_title(title)
        ax.set_xlabel(xcol_t)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xmin_t, xmax_t)
        if legend_title:
            ax.legend(title=legend_title)
        else:
            ax.legend()

    print(f"✅ 📊 Created V plot with {len(ax_v_final.lines)} traces")
    print(f"✅ 📊 Created I plot with {len(ax_i_final.lines)} traces")
    return fig_v_final, ax_v_final, fig_i_final, ax_i_final

# 🟠 📣 [Call Function] - 📊 Final Plots
fig_v_final, ax_v_final, fig_i_final, ax_i_final = create_final_plots(
    dict_df_processed=dict_df_processed,
    legend_label=legend_label,
    plot_title_v=plot_title_v,
    plot_title_i=plot_title_i,
    ycol_v=ycol_v,
    ycol_i=ycol_i,
    xcol_t=xcol_t,
    xmin_t=xmin_t,
    xmax_t=xmax_t,
    legend_title=legend_title
)
# =============================================================================
# 📤 EXPORT FILES
# =============================================================================

def save_all_outputs(
    dict_df_raw,
    dict_df_processed,
    plot_title_v,
    plot_title_i,
    legend_label,
    base_exp_name,
    custom_suffix,
    folder_path_parent,
    fig_v_final,
    fig_i_final,
    ax_v_final,
    ax_i_final,
    xmin_t,
    xmax_t,
    save_formats,
    script_name,
    script_version,
    df_system_info
):

    print("\n🟠 📤 Exporting files...")

    # ---------------------------
    # 📤 📂 CREATE OUTPUT FOLDERS
    # ---------------------------

    # Timestamp for folder naming
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

    # Main output folder
    folder_path_output_main = os.path.join(folder_path_parent, "VT_Py_Outputs")
    folder_name_output_main = os.path.basename(folder_path_output_main)

    # Subfolder with timestamp
    folder_path_output_dated = os.path.join(folder_path_output_main, f"VT_Py_Outputs_DataOverlay_{timestamp_str}")
    folder_name_output_dated = os.path.basename(folder_path_output_dated)

    # Create the folders
    os.makedirs(folder_path_output_main, exist_ok=True)
    os.makedirs(folder_path_output_dated, exist_ok=True)

    print(f"\n✅ 📂 Export folder: {folder_name_output_main} > {folder_name_output_dated}")

    # ---------------------------
    # 📤 🖼️ EXPORT Final Plots (.png, .svg, .eps)
    # ---------------------------
    print("\n")

    # -------- SAVE: Voltage plot --------
    if fig_v_final and ax_v_final.has_data():
    # ax_v.has_data() checks if any data (like lines or points) was plotted.
        file_base_V = f"PyDO_Voltage_wOverlay{custom_suffix}"
        for ext in save_formats:
            fig_v_final.savefig(os.path.join(folder_path_output_dated, f"{file_base_V}.{ext}"))
            print(f"✅ 🖼️ Saved V plot as: {file_base_V}.{ext}")
    else:
        print("🚫 V plot is empty — not saved.")
        
    # -------- SAVE: Current plot --------
    if fig_i_final and ax_i_final.has_data():
        file_base_I = f"PyDO_Current_wOverlay{custom_suffix}"
        for ext in save_formats:
            fig_i_final.savefig(os.path.join(folder_path_output_dated, f"{file_base_I}.{ext}"))
            print(f"✅ 🖼️ Saved I plot as: {file_base_I}.{ext}")
    else:
        print("🚫 I plot is empty — not saved.")

    plt.show()

    # ---------------------------
    # 📤 💾 EXPORT Final TEK (.csv)
    # --------------------------- 

    # -------- SAVE final TEK.CSVs --------
    for file_name_user_offset, df_user_offset in dict_df_processed.items():
        csv_name = f"PyDO_{file_name_user_offset}.csv"
        csv_out_path = os.path.join(folder_path_output_dated, csv_name)
        df_user_offset.to_csv(csv_out_path, index=False)
    print(f"\n✅ 💾 Saved .csv data as: {csv_name}")

    # ---------------------------
    # 📤 📚 EXPORT Dictionaries (.pkl)
    # ---------------------------

    # # ---- SAVE: Raw DataFrames Dictionary ----
    # pickle_name_raw = f"PyDO_Dictionary_AllDataFiles_Raw{custom_suffix}.pkl"
    # pickle_path_raw = os.path.join(folder_path_output_dated, pickle_name_raw)

    # with open(pickle_path_raw, 'wb') as f:
    #     pickle.dump(dict_df_raw, f)
    # print("✅ 📚 Saved .pkl DataFrames Dictionary for raw TEK.csv")

    # ---- SAVE: Offset DataFrames Dictionary ----
    pickle_name_user_offset = f"PyDO_Dictionary_AllDataFiles_Processed{custom_suffix}.pkl"
    pickle_path_user_offset = os.path.join(folder_path_output_dated, pickle_name_user_offset)

    with open(pickle_path_user_offset, 'wb') as f:
        pickle.dump(dict_df_processed, f)
    print(f"✅ 📚 Saved .pkl dictionary as: {pickle_name_user_offset}.csv")

    # ---------------------------
    # 📤 📝 EXPORT PyDO_Summary (.json)
    # ---------------------------

    # 1️⃣ Fill in offset values
    user_offset_summary = {}
    for name_user_offset in dict_df_processed:
        name_original = name_user_offset.replace("_user_offset", "")
        raw_start = dict_df_raw[name_original][xcol_t].min()
        offset_start = dict_df_processed[name_user_offset][xcol_t].min()
        user_offset_val = round(raw_start - offset_start, 6)
        user_offset_summary[name_original] = user_offset_val

    # 2️⃣ Extract subset of rcParams for reproducibility
    rc_subset_keys = [
        "font.family", "font.size",
        "axes.labelsize", "axes.labelweight", "axes.titlesize", "axes.titlepad",
        "axes.linewidth", "axes.facecolor", "axes.grid",
        "xtick.labelsize", "ytick.labelsize", "xtick.major.width", "ytick.major.width",
        "lines.linewidth", "figure.figsize", "savefig.dpi"
    ]
    rcparams_subset = {k: plt.rcParams[k] for k in rc_subset_keys}

    # 3️⃣ Assemble PyDO_Summary dictionary
    PyDO_Summary = {
        "experiment_name": base_exp_name,
        "suffix": custom_suffix,
        "timestamp": timestamp_str,
        "input_folder": folder_path_parent,
        "output_folder": folder_path_output_dated,
        "raw_files": list(dict_df_raw.keys()),
        "xaxis_range_ms": [xmin_t, xmax_t],
        "offset_applied": user_offset_summary,
        "legend_label": legend_label,
        "plot_titles": {"V": plot_title_v, "I": plot_title_i},
        "save_formats": save_formats,
        "n_files_processed": len(dict_df_raw),
        "n_traces": {
            "voltage": len(ax_v_final.lines),
            "current": len(ax_i_final.lines)
        },
        "rcparams_subset": rcparams_subset,
        "script": {
            "name": script_name,
            "version": script_version
        },
        "df_system_info": df_system_info
    }

    # 4️⃣ Save JSON
    json_name = "PyDO_Summary.json"
    json_path = os.path.join(folder_path_output_dated, json_name)

    with open(json_path, 'w') as f:
        json.dump(PyDO_Summary, f, indent=4)

    print(f"✅ 📝 Saved .json {json_name}")

    # ---------------------------
    # 📤 🗒️ EXPORT Plain Text Summary (.txt)
    # ---------------------------

    txt_name = "PyDO_Summary.txt"
    txt_path = os.path.join(folder_path_output_dated, txt_name)

    with open(txt_path, "w") as f:
        f.write("Voltage Transient PyDO_Summary\n")
        f.write("======================================\n\n")

        # 🔹 Script & System Info
        f.write("🛠️ Script & System Info\n")
        f.write("-------------------------\n")
        f.write(f"Script:           {script_name}\n")
        f.write(f"Version:          {script_version}\n")
        f.write(f"User:             {df_system_info['user']}\n")
        f.write(f"System:           {df_system_info['system']} {df_system_info['release']}\n")
        f.write(f"Python:           {df_system_info['python_version']}\n\n")
        
        # 🔹 Experiment Info
        f.write("📌 Experiment Info\n")
        f.write("------------------\n")
        f.write(f"Device Info:      {base_exp_name}\n")
        f.write(f"Experiment Info:  {custom_suffix}\n")
        f.write(f"Timestamp:        {timestamp_str}\n")
        f.write(f"Input Folder:     {folder_path_parent}\n")
        f.write(f"Output Folder:    {folder_path_output_dated}\n\n")
        
            #---------------------------------------------------------------
            # ❗❗❗ To DO: In future, auto-extract the following from directory❗❗❗:
                # Experiment Date
                # Wafer ID
                # Device ID
                # Electrode ID
                # Experiment Type
                # Experiment Info
                    # Current
                    # Pulse Width
                    # IP delay
                    # Frequency
            #---------------------------------------------------------------
        
        # 🔹 File Processing Summary
        f.write("📂 File Processing\n")
        f.write("------------------\n")
        f.write(f"Files Processed:  {len(dict_df_raw)} TEK.CSV files\n")
        f.write(f"Time Range (ms):  {xmin_t} to {xmax_t}\n")
        f.write(f"Traces Plotted:   V={len(ax_v_final.lines)}, I={len(ax_i_final.lines)}\n")
        f.write(f"Plot Titles:      {plot_title_v}, {plot_title_i}\n")
        f.write(f"Save Formats:     {', '.join(save_formats).upper()}\n\n")

        # 🔹 Offset Summary
        f.write("⏱️ Time Offset Summary (ms)\n")
        f.write("-----------------------------\n")
        for name_user_offset in dict_df_processed:
            name_original = name_user_offset.replace("_user_offset", "")
            raw_start = dict_df_raw[name_original][xcol_t].min()
            offset_start = dict_df_processed[name_user_offset][xcol_t].min()
            user_offset_val = raw_start - offset_start
            f.write(f"  • {name_original}: {user_offset_val:.3f} ms\n")
        f.write("\n")

        # 🔹 Legend Labels
        f.write("🏷️ Custom Legend Labels\n")
        f.write("------------------------\n")
        for key, val in legend_label.items():
            f.write(f"  • {key}: {val}\n")
        f.write("\n")
        
        # 🔹 Plotting Config
        f.write("🎨 Plot Style Config (rcParams)\n")
        f.write("-------------------------------\n")
        for k, v in rcparams_subset.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\n")
        
        # 🔹 Exported Files
        f.write("📤 Exported Files\n")
        f.write("------------------\n")

        # Figure files
        for ext in save_formats:
            f.write(f"  • {file_base_V}.{ext}\n")
            f.write(f"  • {file_base_I}.{ext}\n")

        # Processed CSVs
        for file_name_user_offset in dict_df_processed:
            csv_name = f"{base_exp_name}_VTlogs_{file_name_user_offset}.csv"
            f.write(f"  • {csv_name}\n")

        # Pickle files
        # f.write(f"  • {pickle_name_raw}\n")
        f.write(f"  • {pickle_name_user_offset}\n")

        # PyDO_Summary files
        f.write(f"  • {json_name}\n")
        f.write(f"  • {txt_name}\n")

    print("✅ 📝 Saved .txt PyDO_Summary")

    # ---------------------------
    # 📤 📂 Auto-OPEN OUTPUT FOLDER
    # ---------------------------

    # 📂 Automatically open the output folder (if possible)
    try:
        if platform.system() == "Windows":
            os.startfile(folder_path_output_dated)
        elif platform.system() == "Darwin":  # macOS
            os.system(f"open '{folder_path_output_dated}'")
        elif platform.system() == "Linux":
            os.system(f"xdg-open '{folder_path_output_dated}'")
    except Exception as e:
        print(f"🚫 ERROR: Could not open output folder automatically: {e}")
        
    # ---------------------------
    # 📤 📂 Auto-OPEN SUMMARY FILE (.txt)
    # ---------------------------

    def open_summary_txt(txt_path):
        """Attempts to open the summary .txt file based on OS."""
        try:
            system_type = platform.system()
            if system_type == "Windows":
                os.startfile(txt_path)
            elif system_type == "Darwin":  # macOS
                os.system(f"open '{txt_path}'")
            elif system_type == "Linux":
                os.system(f"xdg-open '{txt_path}'")
        except Exception as e:
            print(f"🚫 ERROR: Could not open summary file automatically: {e}")

    # ✅ Call once
    open_summary_txt(txt_path)

# 🟠 📣 [Call Function] - 💾 Save Everything
save_all_outputs(
    dict_df_raw=dict_df_raw,
    dict_df_processed=dict_df_processed,
    plot_title_v=plot_title_v,
    plot_title_i=plot_title_i,
    legend_label=legend_label,
    base_exp_name=base_exp_name,
    custom_suffix=custom_suffix,
    folder_path_parent=folder_path_parent,
    fig_v_final=fig_v_final,
    fig_i_final=fig_i_final,
    ax_v_final=ax_v_final,
    ax_i_final=ax_i_final,
    xmin_t=xmin_t,
    xmax_t=xmax_t,
    save_formats=save_formats,
    script_name=script_name,
    script_version=script_version,
    df_system_info=df_system_info
)

# =============================================================================
# 🌟 END
# =============================================================================

print("\n🌟 CODE COMPLETE — All steps finished successfully! 🎉")
print("_"*80 + "\n")

# =============================================================================
# 📌 🐞 To Do
# =============================================================================
    
# ---------------------------
# RF: 2025-07-09
# ---------------------------

# ✅ PyCode1 for non-annotated plots 
    # Final TEK DataFrame
        # when i make my df_user_offset, create a final dataframe with final plotted files.
        # create and save the df dictionary also using pickle
    # Update code to pull legend and title info based on directory.csv
    # Ask if I want to use default scale limit (-0.1 to 0.7) or define new
    # Create plots > Save. 
    # No annotation from initial plots.
    # Save file name: 
        # ExAMPLE: 
            # 20250616_IF0011_IIA42_VTplots_V_Variable-Current
            # 20250616_IF0011_IIA42_VTplots_I_Variable-Current 
            # 20250616_IF0011_IIA42_VTplots_V_Variable-IPdelay
            # 20250616_IF0011_IIA42_VTplots_I_Variable-IPdelay  
            # 20250616_IF0011_IIA42_VTplots_V_1Billion
            # 20250616_IF0011_IIA42_VTplots_I_1Billion  
    # Rename legends & title based on directory. Even better, Parse info from directory

    # Make the output folder unique per run. Save files in a new sub-folder with date+time stamp in the output folder to avoid confsuion between mutiple versions of the same figure



    # 📌 🟠
    
    # Color cycling / map > Hues (Dark to Light)
    # Plot rectangular box size > consistent for V and I plot
    # User defined variables 
        # > Proof read 
        # > Create dictionary
        # > Include in metadat output summary
    # Legend Label + Title 
        # > Parse from directory
   
    # AUTOMATION: Folder level processing:
        # Part 1: Offset all the TEK.csv files in the folder & save a master TEK dictionary
        # Part 2: Create voltage summary of all files and save Master Voltage Summary Dictionary & Analysis Data Table 
