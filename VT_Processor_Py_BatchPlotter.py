"""
===============================================================================
NeuroEng Voltage Transient Processor - Batch Plotter
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-07-07
Modified:   2025-11-06
Code:       VT_Processor_Py_BatchPlotter
Version:    v1.2.3

DESCRIPTION
---------------------------
- Selects and plots multiple TEK0000x.CSV files using a GUI file picker.
- Standardizes loaded data to canonical Time (µs), Voltage (V), and Current (A).
- Plots Voltage and Current traces on separate, dedicated figures.
- Allows optional time-shifting (offset) to align traces manually.
- Prompts for custom x-axis limits, plot titles, and legend labels via the terminal.
- Saves all plots (PNG/SVG), processed CSVs, and a summary JSON/TXT
  into a standardized, timestamped folder.

OUTPUTS
---------------------------
- **`PyBP_Summary.json` / `.txt`**: JSON and Text files detailing all parameters,
  files, settings, and custom labels used for the run.
- **`PyBP_Dictionary_AllDataFiles_Processed{suffix}.pkl`**: A pickle file
  containing a dictionary where each key holds a DataFrame of the
  (potentially) baseline-corrected and time-offset data.
- **Subfolders** containing the final plots and processed data:
  - **`/PyBP_PNG_Files/`**: Contains the final Voltage vs. Time and
    Current vs. Time plots (e.g., `PyBP_Voltage_{suffix}.png`).
  - **`/PyBP_SVG_Files/`**: Contains plots in scalable vector format.
  - **`/PyBP_CSV_Files/`**: Contains individual CSVs of the final
    processed (and time-offset) data for each trace.
  
===============================================================================
"""

# =============================================================================
# 🧭 SCRIPT ROADMAP (JUMP TO SECTION)
# =============================================================================
# 1. 🛠️ Global Config & Defaults (Imported)
# 2. 📁 File Selection (GUI) & Create Output Folders
# 3. 🔄 Load CSVs + Standardize to µs + Initial Plot Preview
# 4. 🧠 User Inputs: Offset (µs), Axis (µs), Titles, Labels
# 5. 📊 Create Final Plots
# 6. 📤 Export: Plots, CSVs, Pickles, Summary
# 7. 📂 Auto-Open Output Folder & Summary
# 8. 🌟 Completion Message

#==============================================================================
# 🧠 AUTO-DETECT SCRIPT NAME & VERSION (from file name)
#==============================================================================
print("_"*40 + "")
import re 
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
# --- Core Python ---
import time                         # Timing delays and timestamps

# --- Data Handling & Plotting ---
import pandas as pd                 # For CSV reading and data manipulation
import matplotlib
import matplotlib.pyplot as plt     # For plotting
from matplotlib.cm import get_cmap
from matplotlib import cycler
import numpy as np

#--------------------
# 🛠️ IMPORT HELPER MODULES (from .py)
#--------------------

# --- Settings & Constants ---
from VT_Module_Py_GlobalSettings import (
    DICT_SYSTEM_INFO,
    DICT_TEK_CHANNEL_MAP,
    CANON_TIME_UNIT, CANON_TIME, CANON_VOLT, CANON_CURR, # Use canonical names
    # rcParams theme applied automatically on import
)

# --- System Interactions ---
from VT_Module_Py_System import (
    get_script_info,            # Moved from Utilities
    select_csv_files_gui,       # Moved from FileIO
    create_output_structure,    # Moved from Paths
    STAGE_BATCH_PLOTTER,        # Moved from Paths
    print_output_tree,          # Moved from TreeView
    print_latest_output_tree,   # Moved from TreeView
)

# --- Data Input/Output ---
from VT_Module_Py_DataIO import (
    save_pipeline_artifacts,    # Moved from Exporters
)

# --- Data Processing Algorithms ---
from VT_Module_Py_Processing import (
    standardize_dataframe_to_µs, # Moved from GlobalSettings
)

# --- Plotting Helpers ---
from VT_Module_Py_PlotHelpers import (
    place_legend_left, 
    # AUTO_YLABEL_ADJUST and patched_show are applied automatically on import
)

# --- User Input ---
from VT_Module_Py_UserInput import (
    get_user_input,
    get_time_limits_console,
    get_user_offset_values, 
    get_custom_legends,
    get_legend_title, 
    get_plot_titles, 
    get_colormap_choice,
    get_colormap_reverse_with_preview,  # side-by-side preview, returns bool
)

#==============================================================================
# 🛠️ ADDITIONAL CONFIGURATION
#==============================================================================
print("🟠 🛠️ Applying Additional Configuration...")

# Use imported canonical names
xcol_t = CANON_TIME  # "Time (µs)"
ycol_v = CANON_VOLT  # "Voltage (V)"
ycol_i = CANON_CURR  # "Current (A)"

# File input
skip_rows_default = 15
file_picker_title = "Select CSV files"

# Plot titles
plot_title_v_default = "Voltage vs Time"
plot_title_i_default = "Current vs Time"

# Default time axis limits (in µs) 
xaxis_min_default = -100.0  
xaxis_max_default = 700.0  

# Legend
legend_title_default = ""    # Use empty string for "no default"

# File naming
custom_suffix_default = ""

# Save formats: Use 'png' and 'svg' by default, consistent with paths module ---
save_eps_default = False
save_formats = ["png", "svg"] + (["eps"] if save_eps_default else [])

#--------------------
# 🌳 TREE VIEW CONFIG
#--------------------
SHOW_TREE  = True           # False to disable
TREE_MODE  = "current"       # "current", "latest", or "both"

#==============================================================================
# 🛠️ [HELPER FUNCTION]
#==============================================================================

# --------------------
# 🔁 [HELPER FUNCTION] FILE SUFFIX  
# --------------------
SUFFIX_RAW = "_Raw"
SUFFIX_OFFSET = "_UserOffset"

def is_offset_key(name: str) -> bool:
    """Checks if a dictionary key has the _UserOffset suffix.

    Args:
        name (str): The dictionary key or filename string.

    Returns:
        bool: True if the name ends with "_UserOffset" (case-insensitive),
              False otherwise.
    """
    return bool(re.search(r"_UserOffset$", name, flags=re.IGNORECASE))

#==============================================================================
# 🚀 MAIN SCRIPT WORKFLOW
#==============================================================================

def main():
    """Runs the full Batch Plotter pipeline.

    This function wraps the entire workflow, including the nested helper
    functions, to keep the logic outside the global scope. It calls
    `run_batch_plotter` to load data, then gathers all user input,
    and finally calls `create_final_plots` and `save_pipeline_artifacts`.
    """
       
    #====================
    # 🔄 [Helper Fn] LOAD CSV + 📊 INITIAL PLOTS
    #====================
    
    def run_batch_plotter(file_path, skip_rows, xcol_t, ycol_v, ycol_i):
        """Loads CSVs, creates folders, and shows the initial preview.

        This function handles the first half of the workflow:
        1. Prompts user for CSV files (if not provided).
        2. Creates the standardized output folder structure.
        3. Loops through each CSV, loads, and standardizes it.
        4. Generates and displays initial V/I preview plots.
        5. Waits for user to press Enter before continuing.

        Args:
            file_path (str or Path): A direct path to a file or list
                of files to process, skipping the GUI prompt.
            skip_rows (int): The number of header rows to skip.
            xcol_t (str): The canonical name for the time column.
            ycol_v (str): The canonical name for the voltage column.
            ycol_i (str): The canonical name for the current column.

        Returns:
            tuple: A tuple containing:
                (Paths, dict, plt.Figure, plt.Axes, plt.Figure, plt.Axes)
                Specifically: (paths, DICT_DF_Raw, fig_v_init, ax_v_init,
                             fig_i_init, ax_i_init)
        """
        
        #---------------------------
        # 📁 GUI File Picker - Load CSV
        #---------------------------
        # ⭕️ [Helper fn] imported from VT_Module_Py_FileIO.py
        # 📣 [Call fn]
        if not file_path:
            file_paths_list = select_csv_files_gui()
        else:
            # If file_path was passed in, ensure it's a list of Path objects
            file_paths_list = [Path(p) for p in file_path]
            print("  ✅ Using provided file path(s).")
            
        # Base folder where TEK.CSVs came from
        folder_path_parent = file_paths_list[0].parent
        
        # ---------------------------
        # 📤 📂 CREATE EXPORT FOLDERS (using VT_Module_Py_Paths)
        # ---------------------------
        stage_name, stage_id = STAGE_BATCH_PLOTTER
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
    
        # --------------------
        # 🔄 LOAD CSV + 📊 INITIAL PLOTS
        # --------------------
        print("\n🟠 🔄 Loading selected files...")
        DICT_DF_Raw = {}  # key = file_name, value = dataframe
    
        print("🟠 📊 Initializing preview figures...")
        fig_v_init, ax_v_init = plt.subplots(figsize=plt.rcParams["figure.figsize"])
        fig_i_init, ax_i_init = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    
        for csv_path in file_paths_list:
            file_name = csv_path.stem 
            file_selected = csv_path.name
            
            try:
                # Load, then standardize to µs ---
                df_raw_input = pd.read_csv(csv_path, skiprows=skip_rows)
                
                # Use the global standardization function
                df_raw_std = standardize_dataframe_to_µs(df_raw_input, channel_map=DICT_TEK_CHANNEL_MAP)
                
                DICT_DF_Raw[file_name] = df_raw_std
    
                # Plot using standardized columns (µs)
                ax_v_init.plot(df_raw_std[xcol_t], df_raw_std[ycol_v], label=file_name)
                ax_i_init.plot(df_raw_std[xcol_t], df_raw_std[ycol_i], label=file_name)
    
            except Exception as e:
                print(f"🚫 Error reading or standardizing {file_selected}: {e}")
    
        for ax, title, ylabel in zip(
            [ax_v_init, ax_i_init],
            [plot_title_v_default, plot_title_i_default],
            [ycol_v, ycol_i]
        ):
            ax.set_title(title)
            ax.set_xlabel(xcol_t)  # "Time (µs)"
            ax.set_ylabel(ylabel)
            ax.legend()
    
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
    
        plt.show()
    
        if not matplotlib.is_interactive():
            input("🛑 🖼️ Plot shown. 👉 Press ENTER to continue...")
        
        # Return paths object and standardized dict
        return paths, DICT_DF_Raw, fig_v_init, ax_v_init, fig_i_init, ax_i_init
    
    # 🟠 📣 [Call fn] - LOAD CSV + 📊 INITIAL PLOTS
    paths, DICT_DF_Raw, fig_v_init, ax_v_init, fig_i_init, ax_i_init = run_batch_plotter(
        file_path=None,
        skip_rows=skip_rows_default,
        xcol_t=xcol_t,
        ycol_v=ycol_v,
        ycol_i=ycol_i
    )
    
    #==============================================================================
    # 👉 USER INPUT:
    #==============================================================================
    
    # --------------------
    # 👉 ❓ CONTINUE OR RESTART?
    # --------------------
    
    while True:
        print("\n❗ ACTION REQUIRED ❗")
        user_choice = input("👉 ❓ Continue plotting or restart? (Y/N): ").strip().upper()
        if user_choice == 'Y':
            plt.close(fig_v_init)
            plt.close(fig_i_init)
            break
        elif user_choice == 'N':
            print("\n❌ User aborted. ")
            print("🟠 Closing plot windows...")
            print("🟠 Exiting script...")
            print("👉 Note: Plots and Variable Explorer must be cleared manually in IDEs.")
            plt.close('all')    # Close all open plot windows
            print("🚫 Script terminated. Please re-run code.")
            print("_" * 40)
            time.sleep(3)
            raise SystemExit
        else:
            print("🚫 Invalid input. Type Y/N.")
            
            
    #--------------------
    # ⭕️ [Helper fn] imported from VT_Module_Py_UserInput.py
    # 📣 👉 [Call fn] User Inputs
    #--------------------
    
    # 📣 🧭 (Optional) Time Offset per trace 
    DICT_DF_Processed, user_offset_summary_µs = get_user_offset_values(
        DICT_DF_Raw, xcol_t, 
        time_unit=CANON_TIME_UNIT
        )
    
    # 📣 🧭 Custome Time Scale (x-axis limits)
    xmin_t, xmax_t = get_time_limits_console(
        default_min=xaxis_min_default,
        default_max=xaxis_max_default,
        time_unit=CANON_TIME_UNIT
        )
    
    # 📣 🎨 Colormap selection (3×3 gallery saved to PyBP PNG folder)
    cmap_name, inferred_reverse = get_colormap_choice(default_name="viridis", paths=paths)
    cmap_reverse = get_colormap_reverse_with_preview(cmap_name, default_reverse=inferred_reverse)

    # 📣 🏷️ Legend Labels
    DICT_Legend_Label = get_custom_legends(DICT_DF_Processed)
    
    # 📣 🏷️ Legend Title
    legend_title = get_legend_title(legend_title_default)
    
    # 📣 🖊️ Plot Titles
    plot_title_v, plot_title_i = get_plot_titles(plot_title_v_default, plot_title_i_default)
    
    # 📣 🖊️ (Optional)) Filename Suffix
    # Extract base experiment name (e.g., "20250616_IF0011_IIA42")
    folder_name_parent = paths.parent.name
    base_exp_name = folder_name_parent.split("_VT")[0] if "_VT" in folder_name_parent else folder_name_parent
    # Show example output plot names
    print("\n👉 🖊️ Add Suffix to Plot File Names?")
    print("     ➕ Suffix: E01, VarCurrent, VarIPdelay, 1Billion")
    print("     ➕ File Name:  PyBP_Voltage_<suffix>.png")
    
    custom_suffix = get_user_input("   ✏️ Custom Suffix", custom_suffix_default).replace(" ", "-")
    # Add underscore prefix if suffix exists
    custom_suffix = f"_{custom_suffix}" if custom_suffix else ""
    
    if custom_suffix:
        print(f"   ✅ Plot File Name Suffix: {custom_suffix}")
    else:
        print("   ❌ No Suffix")
    
    #==============================================================================
    # 📊 CREATE FINAL PLOTS
    #==============================================================================
    
    def create_final_plots(
        DICT_DF_Processed,
        DICT_Legend_Label,
        plot_title_v,
        plot_title_i,
        ycol_v,
        ycol_i,
        xcol_t,
        xmin_t,
        xmax_t,
        legend_title,
        cmap_name="viridis",
        cmap_reverse=False
    ):
        """Generates the final, formatted V/I plots after user input.

        Takes the processed data and all user-defined settings (offsets,
        labels, titles, axis limits) and generates the two final
        figures for saving.

        Args:
            DICT_DF_Processed (dict): Dict of DataFrames (may be time-offset).
            DICT_Legend_Label (dict): Mapping of file keys to legend labels.
            plot_title_v (str): Title for the voltage plot.
            plot_title_i (str): Title for the current plot.
            ycol_v (str): The canonical name for the voltage column.
            ycol_i (str): The canonical name for the current column.
            xcol_t (str): The canonical name for the time column.
            xmin_t (float): The user-defined X-axis minimum.
            xmax_t (float): The user-defined X-axis maximum.
            legend_title (str): The (optional) title for the legend.

        Returns:
            tuple: A tuple containing the final plot objects:
                (fig_v_final, ax_v_final, fig_i_final, ax_i_final)
        """
        
        print("\n🟠 Generating Final Plots...")
        
        # 🎨 Build color list from the selected colormap
        n_traces = len(DICT_DF_Processed)
        cmap = get_cmap(cmap_name)
        # Create evenly spaced samples from dark→light (viridis low→dark, high→light)
        idx = np.linspace(0.0, 1.0, max(2, n_traces))
        if cmap_reverse:
            idx = idx[::-1]
        colors = [cmap(x) for x in idx]
        
        # 🎨 Apply as color cycle for the two figures we’re about to create
        color_cycle = cycler(color=colors)


        # Initialize plot figures
        fig_v_final, ax_v_final = plt.subplots(figsize=plt.rcParams["figure.figsize"])
        ax_v_final.set_prop_cycle(color_cycle)
        
        fig_i_final, ax_i_final = plt.subplots(figsize=plt.rcParams["figure.figsize"])
        ax_i_final.set_prop_cycle(color_cycle)

    
        # Plot each offset trace
        for file_name, df_user_offset in DICT_DF_Processed.items():
            label = DICT_Legend_Label.get(file_name, file_name)
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
            place_legend_left(ax)
        
            # Axes spines
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
                
        print(f"✅ 📊 Created V plot with {len(ax_v_final.lines)} traces")
        print(f"✅ 📊 Created I plot with {len(ax_i_final.lines)} traces")
        return fig_v_final, ax_v_final, fig_i_final, ax_i_final
    
    # 🟠 📣 [Call fn] - 📊 Final Plots
    fig_v_final, ax_v_final, fig_i_final, ax_i_final = create_final_plots(
        DICT_DF_Processed=DICT_DF_Processed,
        DICT_Legend_Label=DICT_Legend_Label,
        plot_title_v=plot_title_v,
        plot_title_i=plot_title_i,
        ycol_v=ycol_v,
        ycol_i=ycol_i,
        xcol_t=xcol_t,
        xmin_t=xmin_t,
        xmax_t=xmax_t,
        legend_title=legend_title,
        cmap_name=cmap_name, 
        cmap_reverse=cmap_reverse
    )
    
    #==============================================================================
    # 📤 EXPORT FILES
    #==============================================================================
     
    # 1️⃣ Extract subset of rcParams for reproducibility
    rc_subset_keys = [
        "font.family", 
        "font.size",
        "axes.labelsize", 
        "axes.labelweight", 
        "axes.titlesize", 
        "axes.titlepad",
        "axes.linewidth", 
        "axes.facecolor", 
        "axes.grid",
        "xtick.labelsize", 
        "ytick.labelsize", 
        "xtick.major.width", 
        "ytick.major.width",
        "lines.linewidth", 
        "figure.figsize", 
        "savefig.dpi"
        ]
    rcparams_subset = {k: plt.rcParams[k] for k in rc_subset_keys if k in plt.rcParams} 
    
    # 2️⃣ Define what to save 
    figures_to_save = {
        f"PyBP_Voltage{custom_suffix}": fig_v_final,
        f"PyBP_Current{custom_suffix}": fig_i_final
        }
    dataframes_to_save = {
        f"PyBP_{name}.CSV": df 
        for name, df in DICT_DF_Processed.items()
        }
    pickles_to_save = {
        f"PyBP_Dictionary_AllDataFiles_Processed{custom_suffix}.pkl": DICT_DF_Processed
        }
    
    # 3️⃣ Create file lists for the summary from the dicts above
    saved_plots = list(figures_to_save.keys())     # Get the filenames
    saved_csvs = list(dataframes_to_save.keys())
    saved_pickles = list(pickles_to_save.keys())
    
    # 4️⃣ Assemble PyBP_Summary dictionary
    PyBP_Summary = {
        "experiment_name": base_exp_name,
        "suffix": custom_suffix.lstrip('_'), # Remove leading underscore
        "timestamp": paths.timestamp,
        "input_folder": str(paths.parent),  
        "output_folder": str(paths.stage), 
        "raw_files": list(DICT_DF_Raw.keys()),
        "xaxis_range_µs": [xmin_t, xmax_t], 
        "offset_applied_µs": user_offset_summary_µs, 
        "DICT_Legend_Label": DICT_Legend_Label,
        "plot_titles": {"V": plot_title_v, "I": plot_title_i},
        "save_formats": save_formats,
        "n_files_processed": len(DICT_DF_Raw),
        "n_traces": {
            "voltage": len(ax_v_final.lines),
            "current": len(ax_i_final.lines)
        },
        "artifacts": {
            "plots": saved_plots,
            "csvs": saved_csvs,
            "pickles": saved_pickles
        },
        "rcparams_subset": rcparams_subset,
        "colormap": {
            "name": cmap_name,
            "reverse": cmap_reverse,
            "n_traces": len(DICT_DF_Processed)
        },
        "script": {
            "name": script_name,
            "version": script_version
        },
        "system": DICT_SYSTEM_INFO 
    }
    
    # 5️⃣ Make the single call
    # 🟠 📣 [Call fn] - 💾 Save Everything
    save_pipeline_artifacts(
        paths=paths,
        summary_data=PyBP_Summary,
        summary_base_name="PyBP_Summary",
        figures_to_save=figures_to_save,
        dataframes_csv=dataframes_to_save,   
        pickles_stage=pickles_to_save, 
        save_formats=save_formats
    )
    
    #===================== 
    # 🌳 OUTPUT FOLDER TREE (optional)
    #=====================
    if SHOW_TREE:
        try:
            if TREE_MODE in ("current", "both"):
                print("\n🌳 Folder Tree — This Run:")
                print_output_tree(paths.stage)
            if TREE_MODE in ("latest", "both"):
                print_latest_output_tree(paths.output, stage_name=STAGE_BATCH_PLOTTER[0])
        except Exception as e:
            print(f"🚫 Folder tree preview failed: {e}")
        
# =============================================================================
# 🚀 Run
# =============================================================================
if __name__ == "__main__":
    main()

# =============================================================================
# 🌟 END
# =============================================================================
print("_"*40 + "")
print("\n🌟 CODE COMPLETE — All steps finished successfully! 🎉")
print("_"*40 + "\n")
