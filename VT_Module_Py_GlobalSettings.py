"""
===============================================================================
NeuroEng Voltage Transient Processor - Global Settings
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-Oct-17
Updated:    2025-Oct-21
Code:       VT_Module_Py_GlobalSettings.py
Version:    v1.0.0

DESCRIPTION
---------------------------
- Central configuration module for the VT processing pipeline.
- Stores global constants (system info, default Tektronix channel maps).
- Defines a "Single Source of Truth" for canonical column names (e.g., "Time (µs)")
  and plot labels used across all scripts.
- Provides the core `standardize_dataframe_to_µs` function to ingest raw
  CSV data, auto-detect time units (s, ms, µs), map V/I channels, and
  output a clean, standardized DataFrame.
- Applies a global `matplotlib.rcParams` theme for consistent plot styling.

===============================================================================
"""
#--------------------
# 📦 Imports
#--------------------
from __future__ import annotations    # Must be the first non-comment line
import platform
import getpass
from matplotlib import rcParams
import pandas as pd
import numpy as np

#--------------------
# 💻 System Info
#--------------------
DICT_SYSTEM_INFO = {
    "system": platform.system(),
    "release": platform.release(),
    "python_version": platform.python_version(),
    "user": getpass.getuser(),
}

#--------------------
# 🔌 Tektronix Oscilloscope Channel Map (Default))
#--------------------
DICT_TEK_CHANNEL_MAP = {
    "TIME":    "TIME",
    "VOLTAGE": "CH1",   # switch to "CH2" if voltage is on CH2
    "CURRENT": "CH2",   # switch to "CH1" if current is on CH1
}

#--------------------
# 🔁 Column Standardization & 📈 Plot Axis Labels
#--------------------
# 1. Single Source of Truth - Canonical units
CANON_TIME_UNIT = "µs"
CANON_TIME = "Time (µs)"   # Defines the DataFrame column *and* plot label
CANON_VOLT = "Voltage (V)" # Defines the DataFrame column *and* plot label
CANON_CURR = "Current (A)" # Defines the DataFrame column *and* plot label

# 2. Aliases for scripts (derived from source)
xcol_t = CANON_TIME
ycol_v = CANON_VOLT
ycol_i = CANON_CURR

# 3. Dictionary for plotting (derived from source)
DICT_AXIS_LABELS = {
    "TIME":    CANON_TIME,
    "VOLTAGE": CANON_VOLT,
    "CURRENT": CANON_CURR,
}

# 4. Aliases for finding columns in raw TEK files
TIME_ALIASES = {
    "µs": ["Time(µs)", "Time (µs)", "time_µs", "Time_µs", "T (µs)"],
    "us": ["Time(us)", "Time (us)", "time_us", "Time_us", "T (us)"],
    "ms": ["Time(ms)", "Time (ms)", "time_ms", "Time_ms", "T (ms)"],
    "s":  ["Time(s)", "Time (s)", "time_s",  "Time_s",  "T (s)", "TIME", "Time"],
}
VOLT_ALIASES = ["Voltage(V)", "Voltage (V)", "voltage_V","Voltage_V","Voltage","voltage","V","CH1"]
CURR_ALIASES = ["Current(A)","Current (A)","Current","current","current_A","Current_A","I","CH2"]

# Create new lists *without* CH1/CH2 to prioritize channel_map
VOLT_ALIASES_EXPLICIT = [v for v in VOLT_ALIASES if v not in ("CH1", "CH2")]
CURR_ALIASES_EXPLICIT = [c for c in CURR_ALIASES if c not in ("CH1", "CH2")]

def _pick_first_present(df, names):
    """Helper to find the first column name from a list that exists in a DataFrame."""
    for n in names:
        if n in df.columns:
            return n
    return None

def _coerce_numeric(series):
    """Helper to safely convert a Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")

def standardize_dataframe_to_µs(DF_in: pd.DataFrame, channel_map: dict | None = None) -> pd.DataFrame:
    """
    Return a NEW DataFrame with canonical columns: Time (µs), Voltage (V), Current (A)
    - Detects and converts time from s/ms/µs → µs
    - Resolves Voltage/Current by first checking for explicit names (e.g. "Voltage (V)"),
      then using the provided `channel_map` (e.g., {"VOLTAGE": "CH2"}).
    - Coerces to numeric, drops rows with missing canonical cols, and sorts by time.
    """

    df = DF_in.copy()
    # Determine which channel map to use: the one passed in, or the global default
    CHmap = channel_map if channel_map is not None else DICT_TEK_CHANNEL_MAP

    # ---------- TIME ----------
    time_col = None
    scale = 1.0
    for unit_key, aliases in TIME_ALIASES.items():  # unit_key in {"µs","us","ms","s"}
        cand = _pick_first_present(df, aliases)
        if cand:
            time_col = cand
            if unit_key in ("µs", "us"):
                scale = 1.0
            elif unit_key == "ms":
                scale = 1e3
            elif unit_key == "s":
                scale = 1e6
            break
    
    if time_col is None:
        raise KeyError("No recognizable time column found for conversion to µs.")
    
    t_µs = _coerce_numeric(df[time_col]) * scale

    # ---------- VOLTAGE ----------
    # 1. Try to find an explicit name first (e.g., "Voltage (V)")
    v_col = _pick_first_present(df, VOLT_ALIASES_EXPLICIT)
    
    # 2. If no explicit name, use the channel map to find the V channel
    if v_col is None:
        mapped_v = CHmap.get("VOLTAGE") # e.g., "CH1" or "CH2"
        if mapped_v and mapped_v in df.columns:
            v_col = mapped_v
    if v_col is None:
        raise KeyError("No recognizable voltage column found (e.g., 'Voltage (V)' or mapped 'CH1'/'CH2').")
    v = _coerce_numeric(df[v_col])
    
    # ---------- CURRENT ----------
    # 1. Try to find an explicit name first (e.g., "Current (A)")
    i_col = _pick_first_present(df, CURR_ALIASES_EXPLICIT)
    
    # 2. If no explicit name, use the channel map 
    if i_col is None:
        mapped_i = CHmap.get("CURRENT") # e.g., "CH1" or "CH2"
        if mapped_i and mapped_i in df.columns:
            i_col = mapped_i
            
    # 3. Current is optional; only add if found.
    i = _coerce_numeric(df[i_col]) if i_col is not None else None

    # ---------- Assemble canonical DF ----------
    out_data = {CANON_TIME: t_µs, CANON_VOLT: v}
    if i is not None:
        out_data[CANON_CURR] = i
    out = pd.DataFrame(out_data)

    # Clean and sort
    keep_cols = [CANON_TIME, CANON_VOLT] + ([CANON_CURR] if CANON_CURR in out.columns else [])
    out = out.dropna(subset=keep_cols).sort_values(by=CANON_TIME).reset_index(drop=True)
    return out

#--------------------
# 🎨 Global Plot Style (matplotlib.rcParams)
#--------------------
# Apply a consistent theme for all plots
rcParams.update({
    
    # --- 🖋️ Font & Layout ---
    "font.family": "Arial",              # [Arial, Times New Roman, Helvetica, DejaVu Sans]
    "font.size": 10,                     # [Integer = 8, 10, 12]
    "font.weight": "normal",             # [normal, bold, light", medium, heavy]    

    # --- 📐 Axes Style ---
    "axes.labelsize": 12,                # [Integer = 10, 12, 14]
    "axes.labelweight": "normal",        # [normal, bold]
    "axes.labelpad": 4,                  # [Integer] Distance from tick labels

    "axes.titlesize": 14,                # [Integer = 14, 16]
    "axes.titleweight": "bold",          # [normal, bold]
    "axes.titlepad": 10,                 # [Integer/Float] Space b/w title & plot

    "axes.linewidth": 1,                 # Border thickness [Float = 0.8, 1.0, 1.5]
    "axes.facecolor": "white",           # Plot background [white, lightgray, #f0f0f0]
    "axes.grid": False,                  # Gridlines [True/False] 
    # If axes.grid = True     
    "grid.color": "#dddddd",             # [#cccccc, lightgray]
    "grid.linestyle": "--",              # [ - or -- or : or -. ]
    "grid.linewidth": 0.5,               # [Float = 0.3, 0.5, 1.0]

    # --- 📏 Tick Style ---
    "xtick.labelsize": 10,               # [Integer = 8–12]
    "ytick.labelsize": 10,               # [Integer = 8–12]
    "xtick.major.width": 1,              # [Float]
    "ytick.major.width": 1,              # [Float]
    "xtick.direction": "inout",          # [in, out, inout]
    "ytick.direction": "inout",          # [in, out, inout]

    "yaxis.labellocation": "center",
    
    # --- 🗂️ Legend Style ---
    "legend.title_fontsize": 10,         # [Integer]
    "legend.fontsize": 9,                # [Integer]
    "legend.loc": "upper left",          # [best, left, right, center, lower/upper/center left/right/center]
    "legend.frameon": False,             # Show legend box      # True or False
    # If legend.frameon = True
    "legend.edgecolor": "gray",          # Legend box color 
    "legend.fancybox": True,             # Rounded box corners [True/False] 
    "legend.borderpad": 0.4,             # Space around legend box [Float] 

    # --- 📈 Line Style ---
    "lines.linewidth": 1,                # [Float = 1.0, 2.0]
    
    # --- 🖼️ Figure Layout ---
    "figure.figsize": (7.2, 4),          # Plot size [Tuple (width, height) in inches]
    "figure.autolayout": False,          # True = avoids clipping axis labels 
    "figure.constrained_layout.use": True,
    
    # --- 💾 Save Style ---
    "savefig.dpi": 300,                  # Output resolution  [100, 200, 300]
    
    # --- 🔠 Text Rendering ---
    "text.usetex": False,                # Use LaTeX [True (pretty, slow) or False (faster)]
})

# --- Module Exports ---
# Define what gets imported by 'from ... import *'
__all__ = [
    "DICT_SYSTEM_INFO",
    "DICT_TEK_CHANNEL_MAP",
    "CANON_TIME_UNIT",
    "CANON_TIME",
    "CANON_VOLT",
    "CANON_CURR",
    "xcol_t",
    "ycol_v",
    "ycol_i",
    "DICT_AXIS_LABELS",
    "TIME_ALIASES",
    "VOLT_ALIASES",
    "CURR_ALIASES",
    "standardize_dataframe_to_µs",
]