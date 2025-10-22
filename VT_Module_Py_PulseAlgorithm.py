"""
===============================================================================
NeuroEng Voltage Transient Processor - Pulse Algorithm
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-Oct-21
Updated:    2025-Oct-21
Code:       VT_Module_Py_PulseAlgorithm.py
Version:    v1.0.0

DESCRIPTION
---------------------------
- Contains the core biphasic pulse detection algorithm.
- Includes all configurable parameters for the algorithm (thresholds,
  timing, pulse shape, etc.).
- Main function `find_biphasic_window` is imported by Stage 1.
"""
import numpy as np

#===================================
# 📈 PULSE DETECTOR (Biphasic Symmetrical Cathodal First)
#===================================

#--------------------
# 📈 Pulse Detector - User Parameters 
#--------------------

# 💡 Threshold Settings (Finding the Signal)
NOISE_MULT              = 6.0      # MAD noise multiplier
MIN_CURR_THRESHOLD_AMPS = 5e-6     # Fallback Min Current Threshold (Ampere, not time)
FRAC_OF_PEAK            = 0.07     # at least 7% of p99(|I|)

# ⏱️ Pulse Shape & Timing Settings (Defining the "Pulse") (in µs)
MIN_PHASE_WIDTH_µs      = 20       # 20 µs
MIN_PHASE_HOLE_WIDTH_µs = 60       # 60 µs
MAX_INTERPHASE_WIDTH_µs = 500      # 500 µs
PULSE_AMPL_SYMMETRY_BIAS = 0.25     # Amplitude Symmetry Bias (0-1)
PULSE_START_µs          = 0        # Desired pulse start time (µs)
PULSE_PAD_µs            = 200      # Padding (µs) ± Before Pulse Start & After Pulse End
PULSE_OFFSET_µs          = PULSE_START_µs - PULSE_PAD_µs

# Plot Axis Parameters (µs)
XRANGE_EXTRA_µs = 100     # [100 or 200] extra margin beyond the detected pulse on each side
X_MAJOR_TICK_µs   = 100   # major tick spacing in µs
X_MINOR_PER_MAJOR = 2     # number of minor intervals between majors (2 => 50 µs)

#--------------------
# [HELPER FUNCTION] 📈 Pulse Detector 
#--------------------
def mad(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def adaptive_threshold(i):
    abs_i = np.abs(i)
    cutoff = np.quantile(abs_i, 0.5)
    pool = i[abs_i <= cutoff]
    sigma = mad(pool) if pool.size >= 20 else mad(i)
    thr_noise = NOISE_MULT * sigma
    p99 = np.percentile(abs_i, 99)
    thr_peak = FRAC_OF_PEAK * p99
    return max(MIN_CURR_THRESHOLD_AMPS, thr_noise, thr_peak)

def contiguous_regions(mask):
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    jumps = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[jumps + 1]]
    ends   = np.r_[idx[jumps], idx[-1]]
    return list(zip(starts, ends))

def merge_same_sign_lobes(lobes, t, max_hole):
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

def find_biphasic_window(t: np.ndarray, i: np.ndarray) -> tuple[int, int, float] | None:
    """Detect biphasic pulse on i(t).
    Returns (i0 = start_idx, i1 = end_idx, thr = threshold) or None if not found.
    """
    if len(t) < 10:
        return None
    thr = adaptive_threshold(i)
    mask = np.abs(i) > thr
    runs = contiguous_regions(mask)
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
    lobes = merge_same_sign_lobes(lobes, t, MIN_PHASE_HOLE_WIDTH_µs)
    lobes = [L for L in lobes if L["width"] >= MIN_PHASE_WIDTH_µs]
    if not lobes:
        return None

    # pair opposite-signed neighbors
    best_score = -np.inf; s_pair = e_pair = None
    for k in range(len(lobes) - 1):
        L1, L2 = lobes[k], lobes[k+1]
        if L1["sign"] * L2["sign"] >= 0:
            continue
        gap = t[int(L2["s"])] - t[int(L1["e"])]
        if gap < 0 or gap > MAX_INTERPHASE_WIDTH_µs:
            continue
        sym = min(L1["amp"], L2["amp"]) / max(L1["amp"], L2["amp"])
        score = (L1["amp"] + L2["amp"]) * (L1["width"] + L2["width"]) * (PULSE_AMPL_SYMMETRY_BIAS + (1 - PULSE_AMPL_SYMMETRY_BIAS) * sym)
        if score > best_score:
            best_score = score; s_pair, e_pair = int(L1["s"]), int(L2["e"])

    if best_score > -np.inf:
        start_t = max(t[0], t[s_pair] - PULSE_PAD_µs)
        end_t   = min(t[-1], t[e_pair] + PULSE_PAD_µs)
        i0 = int(np.searchsorted(t, start_t, side="left"))
        i1 = int(np.searchsorted(t, end_t,   side="right") - 1)
        i0 = max(0, min(i0, len(t)-1)); i1 = max(0, min(i1, len(t)-1))
        if i1 > i0:
            return i0, i1, thr
    return None

# Export the main function and constants
__all__ = [
    "find_biphasic_window",
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
    # We also export the params needed by the snapshot in the main script
    "XRANGE_EXTRA_µs", 
    "X_MAJOR_TICK_µs", 
    "X_MINOR_PER_MAJOR"
]