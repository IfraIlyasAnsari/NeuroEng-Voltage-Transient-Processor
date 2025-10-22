"""
===============================================================================
NeuroEng Voltage Transient Processor - "TEST" Global Settings
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-10-17
Modified:   2025-10-21
Code:       test_global_settings.py
Version:    v1.0.0

DESCRIPTION
---------------------------
Unit test script for the `VT_Module_Py_GlobalSettings` module.

This script uses `pytest` to verify the robustness and correctness of the
`standardize_dataframe_to_µs` function. It ensures this core function
behaves as expected across a variety of input data conditions.

Test coverage includes:
- **Time Standardization:** Correctly converts time columns from seconds (s),
  milliseconds (ms), and microseconds (µs/us) into the canonical "Time (µs)".
- **Channel Mapping:** Correctly identifies voltage and current columns using
  various aliases (e.Gas, "Voltage (V)", "Current (A)", "CH1", "CH2").
- **Type Coercion:** Successfully converts string-based numerical data
  (e.g., "0.5") into floats.
- **Error Handling:** Properly raises a `KeyError` if essential columns
  (time, voltage, or current) are missing.

Running these tests confirms that the data standardization logic is reliable
and helps prevent data corruption bugs in the processing pipeline.
                                                                                                                           
"""

import numpy as np
import pandas as pd
import pytest

# Try preferred location first; fall back to Stage2 script if needed.
try:
    from VT_Processor_Py_GlobalSettings import (
        standardize_dataframe_to_µs,
        CANON_TIME, CANON_VOLT, CANON_CURR,
    )
except Exception:
    # Fallback: import from your Stage2 module path
    from VT_Processor_PyStage2_VoltageSelector import (  # type: ignore
        standardize_dataframe_to_µs,
        CANON_TIME, CANON_VOLT, CANON_CURR,
    )

@pytest.fixture
def base_vectors():
    # "True" underlying time in microseconds and simple signals
    t_us = np.linspace(0, 9, 10)  # 0..9 µs
    v = np.sin(t_us * 0.1)
    i = np.cos(t_us * 0.1)
    return t_us, v, i

def df_with_headers(time_header, volt_header, curr_header, t_vals, v, i):
    return pd.DataFrame({
        time_header: t_vals,
        volt_header: v,
        curr_header: i,
    })

@pytest.mark.parametrize(
    "time_header, scale",
    [
        ("Time (µs)", 1.0),      # micro sign
        ("Time (us)", 1.0),      # ascii 'us'
        ("Time (ms)", 1e3),      # ms -> µs
        ("Time (s)",  1e6),      # s  -> µs
    ],
)
@pytest.mark.parametrize(
    "volt_header, curr_header",
    [
        ("Voltage (V)", "Current (A)"),
        ("CH1", "CH2"),          # Tek aliases
        ("Voltage", "Current"),  # generic
    ],
)
def test_time_unit_standardization(base_vectors, time_header, scale, volt_header, curr_header):
    t_us, v, i = base_vectors
    # create "raw" time in the chosen unit
    t_in = t_us / scale
    df_raw = df_with_headers(time_header, volt_header, curr_header, t_in, v, i)

    df_std = standardize_dataframe_to_µs(df_raw)

    assert list(df_std.columns) == ["Time (µs)", "Voltage (V)", "Current (A)"]

    # numeric close due to float ops
    np.testing.assert_allclose(df_std["Time (µs)"].to_numpy(), t_us, rtol=0, atol=1e-12)
    np.testing.assert_allclose(df_std["Voltage (V)"].to_numpy(), v, rtol=0, atol=1e-12)
    np.testing.assert_allclose(df_std["Current (A)"].to_numpy(), i, rtol=0, atol=1e-12)

def test_missing_time_raises(base_vectors):
    t_us, v, i = base_vectors
    df = pd.DataFrame({"CH1": v, "CH2": i})
    with pytest.raises(KeyError):
        standardize_dataframe_to_µs(df)

def test_missing_voltage_or_current_raises(base_vectors):
    t_us, v, i = base_vectors
    df = pd.DataFrame({"Time (µs)": t_us, "CH1": v})
    with pytest.raises(KeyError):
        standardize_dataframe_to_µs(df)
    df = pd.DataFrame({"Time (µs)": t_us, "CH2": i})
    with pytest.raises(KeyError):
        standardize_dataframe_to_µs(df)

def test_non_numeric_coercion(base_vectors):
    t_us, v, i = base_vectors
    df = pd.DataFrame({
        "Time (ms)": (t_us / 1e3).astype(str),  # strings, should coerce
        "CH1": pd.Series(v).astype(str),
        "CH2": pd.Series(i).astype(str),
    })
    out = standardize_dataframe_to_µs(df)
    # ensure they were coerced to numeric and scaled
    assert out["Time (µs)"].dtype.kind in "fc"
    assert out["Voltage (V)"].dtype.kind in "fc"
    assert out["Current (A)"].dtype.kind in "fc"
    np.testing.assert_allclose(out["Time (µs)"].to_numpy(), t_us, atol=1e-12)
