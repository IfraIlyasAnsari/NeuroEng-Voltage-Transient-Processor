"""
===============================================================================
NeuroEng Voltage Transient Processor - Module - User Input
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-10-22
Modified:   2025-10-27
Code:       VT_Module_Py_UserInput.py
Version:    v1.0.1

DESCRIPTION
---------------------------
- Centralizes all interactive console prompts for the VT pipeline.
- Includes helper functions for getting:
  - Default-able text/number inputs.
  - Time limits (min/max) for plots.
  - Specific time points (t1-t5) for Stage 2 analysis.
  - Optional time offsets for Batch Plotter traces.
  - Custom legend labels and titles for Batch Plotter plots.
  - Custom plot titles for Batch Plotter plots.
===============================================================================
"""
#==============================================================================
# 📦 Imports
#==============================================================================
import re

import warnings
# Silence Matplotlib’s tight_layout/constrained_layout chatter
warnings.filterwarnings("ignore", message=".*layout has changed to tight.*")
warnings.filterwarnings("ignore", message=".*constrained_layout.*tight_layout.*", category=UserWarning)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tempfile, os, sys
import subprocess
from pathlib import Path

from VT_Module_Py_GlobalSettings import CANON_TIME_UNIT # <-- Needed for get_user_offset_values


# Global display options
VERBOSE_PREVIEW = False   # Set True only if debugging previews

#==============================================================================
# Helper Functions
#==============================================================================

#--------------------
# 🛠️ General Input Helper
#--------------------
def get_user_input(prompt_text: str, default_value: str | float | int) -> str:
    """
    Prompts a user for console input, returning default if Enter is pressed.
    """
    default_str = str(default_value)
    # Handle empty string default case for clarity in prompt
    default_prompt = f"'{default_str}'" if default_str else "[Empty String]"
    
    user_input = input(f"{prompt_text} [Press Enter for {default_prompt}]: ").strip()
    return user_input if user_input else default_str

#--------------------
# 🧭 Time Limit Input (for Plots)
#--------------------
def get_time_limits_console(
        default_min: float, 
        default_max: float, 
        time_unit: str = "µs"
    ) -> tuple[float, float]:
    """
    Prompts user for a min and max time limit for plot X-axes, with defaults.
    """
    print(f"👉 🧭 Set Plot Time Scale ({time_unit}): ")
    try:
        # Use the general helper for each prompt
        xmin_str = get_user_input("    👉 🔽 Minimum Time", default_min)
        xmax_str = get_user_input("    👉 🔼 Maximum Time", default_max)
        xmin = float(xmin_str)
        xmax = float(xmax_str)
        # Basic validation
        if xmax <= xmin:
            print(f"🚫 Max time ({xmax}) must be greater than Min time ({xmin}). Using defaults.")
            xmin, xmax = default_min, default_max
    except ValueError:
        print("🚫 Invalid numeric input. Using default time limits.")
        xmin, xmax = default_min, default_max

    print(f"   ✅ Time Scale set to ({time_unit}): {xmin} to {xmax}")
    return xmin, xmax

#--------------------
# ⏱️ Specific Time Points Input (Stage 2)
#--------------------
def get_user_time_points(defaults: list[float], time_unit: str = "µs") -> list[float]:
    """Prompts the user for 5 specific time points (for Stage 2 analysis)."""
    print(f"\n👉 ⏱️ Enter 5 Time Points ({time_unit}) for Voltage Extraction:")
    time_points_out = []
    # Assumes defaults is a list of 5 numbers
    for i, default_val in enumerate(defaults, 1):
        prompt = f"   ✏️ t{i}"
        try:
            # Use general helper, then convert to float
            t_val_str = get_user_input(prompt, default_val)
            time_points_out.append(float(t_val_str))
        except ValueError:
            print(f"   🚫 Invalid input for t{i}. Using default: {default_val}")
            time_points_out.append(default_val) # Use the original default float
            
    print(f"   ✅ Time points selected ({time_unit}): {time_points_out}")
    return time_points_out

#--------------------
# 🔄 Optional Trace Time Offset Input (Batch Plotter)
#--------------------
def get_user_offset_values(
        DICT_DF_Raw: dict, # Dictionary of DataFrames to potentially offset
        xcol_t: str,       # Name of the time column (e.g., "Time (µs)")
        time_unit: str = CANON_TIME_UNIT # Use unit from GlobalSettings
    ) -> tuple[dict, dict]:
    """
    Asks user if they want to apply time offsets to Batch Plotter traces.
    If yes, prompts for an offset value (in `time_unit`) for each trace.
    Returns:
        DICT_DF_Processed: Dict containing original or time-shifted DataFrames.
        user_offset_summary_µs: Dict mapping original filename to applied offset (0.0 if none).
    """
    DICT_DF_Processed = {}
    user_offset_summary = {} # Store offset in the specified time unit
    
    while True:
        user_choice = input("\n👉 ❓ Apply custom time offset to any traces? (Y/N): ").strip().upper()
        if user_choice in ["Y", "N"]:
            break
        print("   🚫 Invalid input. Please enter Y or N.")

    if user_choice == "Y":
        print(f"\n👉 🖊️ Input Offset Values ({time_unit}):")
        print(f"      💡 Example: Enter -100 to shift a feature currently at 100 {time_unit} to 0 {time_unit}.")
        print("      💡 Enter 0.0 to leave a trace unchanged.")
        for file_name, df_raw in DICT_DF_Raw.items():
            while True:
                prompt = f"   ✏️ Offset for '{file_name}'"
                try:
                    offset_str = get_user_input(prompt, 0.0)
                    user_offset_val = float(offset_str)
                    
                    # Apply offset if non-zero
                    if user_offset_val != 0.0:
                        df_processed = df_raw.copy()
                        # Apply offset directly to the time column
                        df_processed[xcol_t] = df_processed[xcol_t] - user_offset_val
                        # Store the processed DF with a suffix
                        DICT_DF_Processed[file_name + "_UserOffset"] = df_processed
                    else:
                        # Store the original DF with a suffix
                        DICT_DF_Processed[file_name + "_Raw"] = df_raw.copy()
                        
                    user_offset_summary[file_name] = user_offset_val
                    break # Exit inner loop on valid input
                except ValueError:
                    print("   🚫 Invalid numeric input for offset. Please try again.")
                    
        print("\n   ✅ Offset values applied/recorded for all selected traces.")

    else: # user_choice == "N"
        print("   ❌ No custom time offsets will be applied.")
        # Store original DFs with "_Raw" suffix
        for file_name, df_raw in DICT_DF_Raw.items():
            DICT_DF_Processed[file_name + "_Raw"] = df_raw.copy()
            user_offset_summary[file_name] = 0.0 # Record zero offset

    return DICT_DF_Processed, user_offset_summary

#--------------------
# 🏷️ Legend Label Helpers (Batch Plotter)
#--------------------

# --- Moved here from BatchPlotter ---
def strip_processed_suffix(name: str) -> str:
    """Return the original base name by removing _Raw or _UserOffset."""
    # Use regex to remove suffix, ignoring case
    return re.sub(r"_(Raw|UserOffset)$", "", name, flags=re.IGNORECASE)

def get_custom_legends(DICT_DF_Processed: dict) -> dict:
    """Prompts user for custom legend labels for Batch Plotter traces."""
    DICT_Legend_Label = {}
    print("\n👉 🖊️ Input Legend Labels for Plots:")
    print("      💡 Press Enter to use the base filename (e.g., 'TEK00001').")
    
    # Iterate through the keys of the processed dictionary (which might have suffixes)
    for file_key in DICT_DF_Processed:
        # Get the original base name for the prompt
        base_name = strip_processed_suffix(file_key) 
        prompt = f"   ✏️ Label for '{base_name}'"
        
        # Use base_name as default if user presses Enter
        label = get_user_input(prompt, base_name) 
        
        # Store the chosen label using the potentially suffixed key
        DICT_Legend_Label[file_key] = label 
        
    print("\n   ✅ Legend labels recorded for all traces.")
    return DICT_Legend_Label

def get_legend_title(default_title: str = "") -> str:
    """Prompts user for an optional legend title for Batch Plotter."""
    print("\n👉 🖊️ Input Optional Legend Title:")
    legend_title = get_user_input("   ✏️ Legend Title", default_title)
    if legend_title:
        print(f"   ✅ Legend Title set to: '{legend_title}'")
    else:
        print("   ❌ No legend title will be used.")
    return legend_title

#--------------------
# ✏️ Plot Title Input (Batch Plotter)
#--------------------
def get_plot_titles(default_v: str, default_i: str) -> tuple[str, str]:
    """Prompts user for custom plot titles for Batch Plotter V and I plots."""
    print ("\n👉 🖊️ Input Plot Titles:")
    plot_title_v = get_user_input("   ✏️ Title for Voltage plot", default_v)
    plot_title_i = get_user_input("   ✏️ Title for Current plot", default_i)
    print(f"   ✅ Plot Titles set: V='{plot_title_v}', I='{plot_title_i}'")
    return plot_title_v, plot_title_i

# --------------------
# 🎨 Colormap choice (Batch Plotter)
# --------------------
def get_colormap_choice(default_name: str = "viridis", paths=None) -> tuple[str, bool]:
    """Prompt user for a colormap and save preview in PyBP PNG folder (or temp)."""
    
    def _os_open(path: str):
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", path], check=False)
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception:
            pass

    def _has_gui_backend() -> bool:
        b = matplotlib.get_backend().lower()
        return any(k in b for k in ["qt", "tkagg", "macosx", "wxagg"])

    grid_names = [
        "viridis", "viridis_r", "cividis",
        "cividis_r", "plasma", "plasma_r",
        "tab10", "Set2", "Paired",
    ]

    print("\n🎨 Choose a colormap  [viridis, cividis, plasma, tab10, Set2, Paired]")
    print("   💡 'viridis' → color-blind friendly dark→light gradient")

    # --- Build preview figure ---
    n_rows, n_cols, n_colors = 3, 3, 6
    x = np.linspace(0, 10, 200)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 9), sharey=True)
    try: fig.set_constrained_layout(False)
    except Exception: pass
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Colormap Gallery — normal & reversed (reference)", fontsize=13, y=0.995)

    for ax, cmap_name in zip(axes.flat, grid_names):
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        for i, c in enumerate(colors):
            ax.plot(x, np.sin(x + i * 0.4), color=c, linewidth=2)
        ax.set_title(cmap_name, fontsize=10, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_alpha(0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Save preview ---
    outfile_name = "PyBP_Colormap_Preview.png"
    saved_path = None
    try:
        if paths is not None and getattr(paths, "png", None):
            target_dir = Path(paths.png)
            target_dir.mkdir(parents=True, exist_ok=True)
            saved_path = target_dir / outfile_name
        else:
            tmp = tempfile.NamedTemporaryFile(prefix="colormap_gallery_", suffix=".png", delete=False)
            tmp.close()
            saved_path = Path(tmp.name)
        fig.savefig(saved_path, dpi=160, bbox_inches="tight")
    except Exception:
        saved_path = None

    # --- Show one preview only ---
    if _has_gui_backend():
        try:
            plt.show(block=False); plt.pause(0.6)
        except Exception:
            pass
        print("   📷 Preview ready.")
    else:
        if saved_path is not None:
            print("   📷 Preview ready.")
            _os_open(str(saved_path))

    # --- User input ---
    name_raw = get_user_input("   👉 Colormap name", default_name).strip()
    try: plt.close(fig)
    except Exception: pass

    # --- Validate and infer reverse ---
    def _validate(name_in: str) -> tuple[str, bool]:
        try:
            plt.get_cmap(name_in)
        except Exception:
            print(f"   ⚠️ '{name_in}' not found. Using default '{default_name}'.")
            name_in = default_name
        inferred_rev = False
        clean = name_in
        if name_in.endswith("_r"):
            clean = name_in[:-2]; inferred_rev = True
            try: plt.get_cmap(clean)
            except Exception:
                clean = name_in; inferred_rev = False
        return clean, inferred_rev

    return _validate(name_raw)


# --------------------
# 🎨 Reverse Preview
# --------------------
def get_colormap_reverse_with_preview(cmap_name: str, default_reverse: bool = False) -> bool:
    """Show side-by-side normal vs reversed preview, then ask Y/N."""
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import tempfile, os, sys
    from pathlib import Path

    def _os_open(path: str):
        try:
            if sys.platform.startswith("darwin"):
                import subprocess; subprocess.run(["open", path], check=False)
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                import subprocess; subprocess.run(["xdg-open", path], check=False)
        except Exception:
            pass

    def _has_gui_backend() -> bool:
        b = matplotlib.get_backend().lower()
        return any(k in b for k in ["qt", "tkagg", "macosx", "wxagg"])

    n_colors = 6
    x = np.linspace(0, 10, 200)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    try: fig.set_constrained_layout(False)
    except Exception: pass
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"Reverse Preview: {cmap_name} vs {cmap_name}_r", fontsize=12, y=1.02)

    for ax, name in zip(axes, (cmap_name, f"{cmap_name}_r")):
        try: cmap = plt.get_cmap(name)
        except Exception: cmap = plt.get_cmap("viridis")
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        for i, c in enumerate(colors):
            ax.plot(x, np.cos(x + i * 0.5), color=c, linewidth=2)
        ax.set_title(name, fontsize=10, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_alpha(0.5)

    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(prefix=f"reverse_preview_{cmap_name}_", suffix=".png", delete=False)
    tmp.close(); saved_path = Path(tmp.name)
    fig.savefig(saved_path, dpi=160, bbox_inches="tight")

    if _has_gui_backend():
        try:
            plt.show(block=False); plt.pause(0.6)
        except Exception:
            pass
        print("   📷 Reverse preview ready.")
    else:
        print("   📷 Reverse preview ready.")
        _os_open(str(saved_path))

    while True:
        ans = input(f"   ✏️ Reverse colormap order? (Y/N) [default={'Y' if default_reverse else 'N'}]: ").strip().upper()
        if ans == "":
            choice = default_reverse; break
        if ans in ("Y", "N"):
            choice = (ans == "Y"); break
        print("   🚫 Invalid input. Please enter Y or N.")

    try: plt.close(fig)
    except Exception: pass
    return choice

#==============================================================================
# 📤 Module Exports
#==============================================================================
# Define what helpers gets imported by 'from VT_Module_Py_UserInput import *'

__all__ = [
    # General
    "get_user_input",
    # Specific Prompts
    "get_time_limits_console",
    "get_user_time_points",
    "get_user_offset_values",
    "get_custom_legends",
    "get_legend_title",  
    "get_plot_titles",
    "strip_processed_suffix", 
    "get_colormap_choice", # (returns tuple)
    "get_colormap_reverse_with_preview", 
]

# Helpers used within this module