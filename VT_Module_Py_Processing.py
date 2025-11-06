"""
===============================================================================
NeuroEng Voltage Transient Processor - Data Processing Algorithms
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-Oct-27
Modified:   2025-Oct-27
Code:       VT_Module_Py_Processing.py
Version:    v1.0.0

DESCRIPTION
---------------------------
- Contains core data processing algorithms for the VT pipeline.
- Includes the `standardize_dataframe_to_µs` function for ingesting raw
  Tektronix CSV data, converting units, mapping channels, and outputting
  a clean DataFrame with canonical column names.
- Includes the `find_biphasic_window` function (and its helpers/constants)
  for detecting biphasic pulses in current traces (used in Stage 1).

NOTE
---------------------------
- Global constants and settings are in VT_Module_Py_GlobalSettings.py.
- Data I/O (reading/writing files) is in VT_Module_Py_DataIO.py.
===============================================================================
"""
#==============================================================================
# 📦 Imports
#==============================================================================
from __future__ import annotations # Must be first non-comment line
import numpy as np
import pandas as pd
from typing import List, Tuple

# Import constants and canonical names from GlobalSettings
from VT_Module_Py_GlobalSettings import (
    DICT_TEK_CHANNEL_MAP, # Default channel map
    CANON_TIME,           # e.g., "Time (µs)"
    CANON_VOLT,           # e.g., "Voltage (V)"
    CANON_CURR            # e.g., "Current (A)"
)
#==============================================================================
# 🔄 DATAFRAME STANDARDIZATION 
#==============================================================================
# --- Column Name Aliases (for finding columns in raw Tektronix files) ---
# Moved here from GlobalSettings
TIME_ALIASES = {
    "µs": ["Time(µs)", "Time (µs)", "time_µs", "Time_µs", "T (µs)"], # Microseconds (preferred)
    "us": ["Time(us)", "Time (us)", "time_us", "Time_us", "T (us)"], # Microseconds (alternative)
    "ms": ["Time(ms)", "Time (ms)", "time_ms", "Time_ms", "T (ms)"], # Milliseconds
    "s":  ["Time(s)", "Time (s)", "time_s",  "Time_s",  "T (s)", "TIME", "Time"], # Seconds (base unit)
}
# Includes common variations and generic CH1/CH2
VOLT_ALIASES = ["Voltage(V)", "Voltage (V)", "voltage_V","Voltage_V","Voltage","voltage","V","CH1"]
CURR_ALIASES = ["Current(A)","Current (A)","Current","current","current_A","Current_A","I","CH2"]

# Create versions *without* CH1/CH2 to prioritize explicit names over channel mapping
VOLT_ALIASES_EXPLICIT = [v for v in VOLT_ALIASES if v not in ("CH1", "CH2")]
CURR_ALIASES_EXPLICIT = [c for c in CURR_ALIASES if c not in ("CH1", "CH2")]

# --- Helper Functions for Standardization ---
# Moved here from GlobalSettings
def _pick_first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    """Finds the first column name from `names` that exists in `df.columns`."""
    for n in names:
        if n in df.columns:
            return n
    return None # Return None if none of the names are found

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Safely converts a Pandas Series to numeric, forcing errors into NaN."""
    # errors='coerce' turns unparseable values (like text) into Not-a-Number (NaN)
    return pd.to_numeric(series, errors="coerce")

# --- Main Standardization Function ---
# Moved here from GlobalSettings
def standardize_dataframe_to_µs(
        DF_in: pd.DataFrame, 
        channel_map: dict | None = None
    ) -> pd.DataFrame:
    """
    Takes a raw DataFrame (from Tektronix CSV) and returns a NEW, standardized 
    DataFrame with canonical columns: "Time (µs)", "Voltage (V)", "Current (A)".

    Steps:
    1. Detects time column using aliases and converts units to microseconds (µs).
    2. Finds Voltage column: first by explicit aliases (e.g., "Voltage(V)"), 
       then using the channel_map (e.g., {"VOLTAGE": "CH1"}).
    3. Finds Current column (optional): similarly, checks explicit aliases first, 
       then uses the channel_map.
    4. Converts identified columns to numeric data types (errors become NaN).
    5. Creates a new DataFrame with only the canonical columns.
    6. Drops rows where Time or Voltage is NaN.
    7. Sorts the DataFrame by Time.
    8. Resets the index.
    
    Args:
        DF_in: The input DataFrame read from the CSV.
        channel_map: Optional dictionary mapping {"VOLTAGE": "CHx", "CURRENT": "CHy"}.
                     If None, uses DICT_TEK_CHANNEL_MAP from GlobalSettings.

    Returns:
        A new DataFrame with standardized columns and units.

    Raises:
        KeyError: If a recognizable time or voltage column cannot be found.
    """

    df = DF_in.copy() # Work on a copy to avoid modifying the original DataFrame
    
    # Use provided channel map, or fall back to the global default
    CHmap = channel_map if channel_map is not None else DICT_TEK_CHANNEL_MAP

    # ---------- 1. Find and Convert Time Column ----------
    time_col_found = None
    time_scale_factor = 1.0 # Default scale (assumes µs)
    
    # Iterate through units (µs, ms, s) and their aliases
    for unit_key, aliases in TIME_ALIASES.items():
        potential_col = _pick_first_present(df, aliases)
        if potential_col:
            time_col_found = potential_col
            # Set scale factor based on detected unit
            if unit_key == "ms": time_scale_factor = 1e3  # ms to µs
            elif unit_key == "s": time_scale_factor = 1e6 # s to µs
            # µs or us needs scale factor 1.0 (already set)
            break # Stop searching once a time column is found
    
    if time_col_found is None:
        raise KeyError("Standardization failed: No recognizable time column found in input DataFrame. "
                       f"Searched for aliases: {list(TIME_ALIASES.values())}")
                       
    # Apply scaling and coerce to numeric
    t_µs = _coerce_numeric(df[time_col_found]) * time_scale_factor

    # ---------- 2. Find Voltage Column ----------
    v_col_found = None
    # First, try explicit names like "Voltage (V)"
    v_col_found = _pick_first_present(df, VOLT_ALIASES_EXPLICIT)
    
    # If not found, use the channel map (e.g., find "CH1" if mapped to VOLTAGE)
    if v_col_found is None:
        mapped_v_channel = CHmap.get("VOLTAGE") # Get channel name like "CH1"
        if mapped_v_channel and mapped_v_channel in df.columns:
            v_col_found = mapped_v_channel
            
    if v_col_found is None:
        raise KeyError("Standardization failed: No recognizable voltage column found. "
                       f"Searched for explicit names: {VOLT_ALIASES_EXPLICIT}, "
                       f"and checked channel map key 'VOLTAGE' (mapped to: {CHmap.get('VOLTAGE')}).")
                       
    # Coerce voltage column to numeric
    v = _coerce_numeric(df[v_col_found])
    
    # ---------- 3. Find Current Column (Optional) ----------
    i_col_found = None
    # First, try explicit names like "Current (A)"
    i_col_found = _pick_first_present(df, CURR_ALIASES_EXPLICIT)
    
    # If not found, use the channel map
    if i_col_found is None:
        mapped_i_channel = CHmap.get("CURRENT") # Get channel name like "CH2"
        if mapped_i_channel and mapped_i_channel in df.columns:
            i_col_found = mapped_i_channel
            
    # Coerce current column to numeric *only if found*
    i = _coerce_numeric(df[i_col_found]) if i_col_found is not None else None

    # ---------- 4. Assemble, Clean, and Sort Output DataFrame ----------
    output_data = {
        CANON_TIME: t_µs, 
        CANON_VOLT: v
    }
    # Add current column only if it was successfully found and processed
    if i is not None:
        output_data[CANON_CURR] = i
        
    df_standardized = pd.DataFrame(output_data)

    # --- Data Cleaning ---
    # Define columns essential for a valid row (Time and Voltage are required)
    required_cols = [CANON_TIME, CANON_VOLT] 
    # Drop rows where any of these required columns have NaN values
    df_cleaned = df_standardized.dropna(subset=required_cols) 
    
    # Sort by the canonical time column
    df_sorted = df_cleaned.sort_values(by=CANON_TIME)
    
    # Reset the DataFrame index to be sequential (0, 1, 2, ...) after sorting/dropping
    df_final = df_sorted.reset_index(drop=True) 
    
    return df_final

#==============================================================================
# 📈 PULSE DETECTION ALGORITHM 
#==============================================================================

#--------------------
# Algorithm Constants & Parameters
#--------------------
# Moved here from VT_Module_Py_PulseAlgorithm.py

# Threshold Settings (for distinguishing signal from noise)
NOISE_MULT                  = 6.0      # Multiplier for Median Absolute Deviation (MAD) noise estimation
MIN_CURR_THRESHOLD_AMPS     = 5e-6     # Absolute minimum current (A) to be considered signal
FRAC_OF_PEAK                = 0.07     # Threshold relative to the 99th percentile of absolute current

# Pulse Shape & Timing Settings (defining what constitutes a "biphasic pulse")
# All time values should be in microseconds (µs) consistent with CANON_TIME_UNIT
MIN_PHASE_WIDTH_µs          = 20.0     # Minimum duration (µs) for a single phase (lobe) to be valid
MIN_PHASE_HOLE_WIDTH_µs     = 60.0     # Max duration (µs) of a gap *within* a phase to be bridged
MAX_INTERPHASE_WIDTH_µs     = 500.0    # Max duration (µs) of the gap *between* the two phases
PULSE_AMPL_SYMMETRY_BIAS    = 0.25     # Weighting factor (0-1) for amplitude symmetry in scoring pairs 
                                       # (0 = no bias, 1 = only amplitude matters)

# Windowing Settings (defining the output time window around the detected pulse)
PULSE_START_µs              = 0.0      # Target start time (µs) for the *output* window in Stage 1 offset data
PULSE_PAD_µs                = 200.0    # Padding (µs) added before the detected pulse start and after its end
                                       # Calculated offset: how much to shift time so padded window starts at PULSE_START_µs - PULSE_PAD_µs
PULSE_OFFSET_µs             = PULSE_START_µs - PULSE_PAD_µs 

# Plotting Parameters (used by Stage 1 for diagnostic plots)
XRANGE_EXTRA_µs             = 100.0    # Extra margin (µs) added to each side of the plot's x-axis
X_MAJOR_TICK_µs             = 100.0    # Spacing (µs) for major x-axis ticks
X_MINOR_PER_MAJOR           = 2        # Number of minor intervals per major tick (e.g., 2 means minor ticks every 50 µs if major is 100 µs)

#--------------------
# Algorithm Helper Functions 
#--------------------
# Moved here from VT_Module_Py_PulseAlgorithm.py

def _mad(x: np.ndarray) -> float:
    """Calculate Median Absolute Deviation (scaled to estimate standard deviation)."""
    # Ensure input is a numpy array
    arr = np.asarray(x)
    # Calculate the median of the absolute deviations from the median
    med = np.median(arr)
    # Scale factor 1.4826 makes MAD comparable to standard deviation for normal distribution
    return 1.4826 * np.median(np.abs(arr - med))

def _adaptive_threshold(i: np.ndarray) -> float:
    """Calculate an adaptive threshold based on noise level and peak signal."""
    abs_i = np.abs(i)
    # Estimate noise from the lower 50% of the signal magnitude
    cutoff = np.quantile(abs_i, 0.5)
    # Only use low-signal pool if it's reasonably large
    noise_pool = i[abs_i <= cutoff]
    # Calculate noise sigma using MAD
    sigma = _mad(noise_pool) if noise_pool.size >= 20 else _mad(i) 
    # Noise-based threshold
    thr_noise = NOISE_MULT * sigma
    # Peak-based threshold (robust against single spikes using 99th percentile)
    p99 = np.percentile(abs_i, 99)
    thr_peak = FRAC_OF_PEAK * p99
    # Final threshold is the max of noise-based, peak-based, and absolute minimum
    return max(MIN_CURR_THRESHOLD_AMPS, thr_noise, thr_peak)

def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find start and end indices of contiguous True regions in a boolean mask."""
    if not mask.any(): return [] # Return empty list if no True values
    
    # Find indices where mask is True
    idx = np.flatnonzero(mask)
    if idx.size == 0: return []
    
    # Find indices where the difference between consecutive indices is > 1 (indicates a gap)
    jumps = np.where(np.diff(idx) > 1)[0]
    
    # Start indices: the first index, plus the index after each jump
    starts = np.concatenate(([idx[0]], idx[jumps + 1]))
    # End indices: the index before each jump, plus the last index
    ends = np.concatenate((idx[jumps], [idx[-1]]))
    
    # Pair starts and ends
    return list(zip(starts, ends))

def _merge_same_sign_lobes(lobes: list[dict], t: np.ndarray, max_hole_time: float) -> list[dict]:
    """Merges adjacent lobes of the same sign if the gap between them is small."""
    if not lobes: return []
    
    merged = [lobes[0].copy()] # Start with the first lobe
    for current_lobe in lobes[1:]:
        prev_lobe = merged[-1]
        # Calculate time gap between the end of the previous lobe and start of the current one
        time_gap = t[int(current_lobe["s"])] - t[int(prev_lobe["e"])]
        
        # Check if signs match and the gap is within the allowed maximum "hole" time
        if (current_lobe["sign"] == prev_lobe["sign"]) and (0 <= time_gap <= max_hole_time):
            # Merge: update end index, recalculate width, keep the larger amplitude
            prev_lobe["e"] = int(current_lobe["e"])
            prev_lobe["width"] = t[prev_lobe["e"]] - t[prev_lobe["s"]]
            # Take the max of the absolute amplitudes as representative
            prev_lobe["amp"] = max(prev_lobe["amp"], current_lobe["amp"]) 
        else:
            # No merge possible, add the current lobe as a new entry
            merged.append(current_lobe.copy())
            
    return merged

# --- Main Pulse Finding Function ---
# Moved here from VT_Module_Py_PulseAlgorithm.py
def find_biphasic_window(
        t: np.ndarray, 
        i: np.ndarray
    ) -> Tuple[int, int, float] | None:
    """
    Detects the most prominent biphasic pulse in the current trace i(t).
    Assumes time `t` is in microseconds (µs).

    Algorithm:
    1. Calculate an adaptive threshold based on noise and signal peak.
    2. Find contiguous regions ("runs") where absolute current exceeds the threshold.
    3. Characterize each run as a "lobe" (start/end index, width, sign, median amplitude).
    4. Merge adjacent lobes of the *same* sign if separated by a small gap (<= MIN_PHASE_HOLE_WIDTH_µs).
    5. Filter out merged lobes that are too short (width < MIN_PHASE_WIDTH_µs).
    6. Iterate through pairs of *adjacent, opposite-signed* lobes.
    7. Score each valid pair based on amplitude, width, and amplitude symmetry, penalizing large inter-phase gaps.
    8. Select the pair with the highest score.
    9. Define the window start/end indices by adding padding (PULSE_PAD_µs) to the best pair's time span.
    10. Clip window indices to array bounds.

    Returns:
        Tuple (start_index, end_index, threshold) if a biphasic pulse is found.
        None otherwise.
    """
    # Basic input validation
    if len(t) < 10 or len(i) < 10 or len(t) != len(i):
        # print("   ⚠️ find_biphasic_window: Input arrays too short or mismatched length.")
        return None
        
    # 1. Calculate threshold
    threshold = _adaptive_threshold(i)
    
    # 2. Find regions above threshold
    mask = np.abs(i) > threshold
    runs = _contiguous_regions(mask)
    if not runs: 
        # print("   ℹ️ find_biphasic_window: No regions found above threshold.")
        return None

    # 3. Characterize lobes
    lobes = []
    for s_idx, e_idx in runs:
        # Ensure indices are integers
        s, e = int(s_idx), int(e_idx)
        # Calculate width in time units (µs)
        width = t[e] - t[s]
        # Skip if width is non-positive (can happen with single points)
        if width <= 0: continue
        
        # Extract the current segment for this lobe
        segment = i[s : e + 1] # Include end index
        if segment.size == 0: continue # Should not happen if width > 0, but safety check
        
        # Use median of absolute values as representative amplitude (robust to spikes)
        amplitude = float(np.median(np.abs(segment)))
        # Determine sign based on median of the segment (more robust than mean)
        # If median is exactly zero, default to positive sign (arbitrary but consistent)
        sign = float(np.sign(np.median(segment))) or 1.0 
        
        lobes.append({"s": s, "e": e, "amp": amplitude, "sign": sign, "width": width})
        
    if not lobes: 
        # print("   ℹ️ find_biphasic_window: No valid lobes created from regions.")
        return None

    # 4. Merge small gaps within phases
    lobes = _merge_same_sign_lobes(lobes, t, MIN_PHASE_HOLE_WIDTH_µs)
    
    # 5. Filter out short lobes
    lobes = [L for L in lobes if L["width"] >= MIN_PHASE_WIDTH_µs]
    if len(lobes) < 2: # Need at least two lobes for a biphasic pulse
        # print("   ℹ️ find_biphasic_window: Fewer than 2 valid lobes remain after merging/filtering.")
        return None

    # 6. Score adjacent, opposite-signed pairs
    best_score = -np.inf
    best_pair_indices = None # Store (start_idx_L1, end_idx_L2)

    for k in range(len(lobes) - 1):
        L1, L2 = lobes[k], lobes[k+1]
        
        # Check for opposite signs
        if L1["sign"] * L2["sign"] >= 0: continue 
        
        # Check inter-phase gap duration
        interphase_gap = t[int(L2["s"])] - t[int(L1["e"])]
        if not (0 <= interphase_gap <= MAX_INTERPHASE_WIDTH_µs): continue
            
        # Calculate amplitude symmetry ratio (value between 0 and 1)
        amp_sym_ratio = min(L1["amp"], L2["amp"]) / max(L1["amp"], L2["amp"]) if max(L1["amp"], L2["amp"]) > 0 else 1.0
        
        # Calculate score: combines total amplitude, total width, and weighted symmetry
        # Higher amplitude, larger width, and better symmetry increase the score.
        score = (L1["amp"] + L2["amp"]) * \
                (L1["width"] + L2["width"]) * \
                (PULSE_AMPL_SYMMETRY_BIAS + (1 - PULSE_AMPL_SYMMETRY_BIAS) * amp_sym_ratio)
                
        # Optional: Penalize larger gaps slightly? (Could add term like `* (1 - gap / MAX_INTERPHASE_WIDTH_µs)`)
                
        # Update best score and indices if current pair is better
        if score > best_score:
            best_score = score
            best_pair_indices = (int(L1["s"]), int(L2["e"]))

    # 7. Define window based on best pair (if found)
    if best_pair_indices:
        s_pair_idx, e_pair_idx = best_pair_indices
        
        # Calculate target start and end times with padding
        target_start_time = t[s_pair_idx] - PULSE_PAD_µs
        target_end_time   = t[e_pair_idx] + PULSE_PAD_µs
        
        # Find corresponding indices in the time array using searchsorted
        # 'left' side finds first index >= target_start_time
        # 'right' side finds first index > target_end_time, so subtract 1 for inclusive end
        start_idx = int(np.searchsorted(t, target_start_time, side="left"))
        end_idx   = int(np.searchsorted(t, target_end_time,   side="right")) - 1
        
        # --- Clipping to array bounds ---
        # Ensure indices are within the valid range [0, len(t)-1]
        start_idx_clipped = max(0, start_idx)
        end_idx_clipped   = min(len(t) - 1, end_idx)
        
        # Final check: Ensure the window is valid (at least 2 points)
        if end_idx_clipped > start_idx_clipped:
            # print(f"   ✅ find_biphasic_window: Found pulse. Score={best_score:.2f}. Window indices: [{start_idx_clipped}, {end_idx_clipped}]")
            return start_idx_clipped, end_idx_clipped, threshold
            
    # If no valid pair was found or window is invalid
    # print("   ℹ️ find_biphasic_window: No suitable biphasic pair found.")
    return None

#==============================================================================
# 📤 Module Exports
#==============================================================================
# Define what gets imported by 'from VT_Module_Py_Processing import *'

__all__ = [
    # Main Functions
    "standardize_dataframe_to_µs",
    "find_biphasic_window",
    
    # Pulse Algorithm Constants (exported for potential use/logging in main scripts)
    "NOISE_MULT",
    "MIN_CURR_THRESHOLD_AMPS",
    "FRAC_OF_PEAK",
    "MIN_PHASE_WIDTH_µs",
    "MIN_PHASE_HOLE_WIDTH_µs",
    "MAX_INTERPHASE_WIDTH_µs",
    "PULSE_AMPL_SYMMETRY_BIAS",
    "PULSE_START_µs",
    "PULSE_PAD_µs",
    "PULSE_OFFSET_µs",
    
    # Plotting Parameters (exported for Stage 1 diagnostic plots)
    "XRANGE_EXTRA_µs",
    "X_MAJOR_TICK_µs",
    "X_MINOR_PER_MAJOR"
]