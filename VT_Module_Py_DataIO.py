"""
===============================================================================
NeuroEng Voltage Transient Processor - Module - Data Input/Output
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-10-22
Modified:   2025-11-11
Code:       VT_Module_Py_DataIO.py
Version:    v1.3.3  

DESCRIPTION
---------------------------
- Centralizes all data file Input/Output operations for the VT pipeline.
[... existing code ...]
- Provides a function `process_metadata_snapshot` for calculating
  derived metadata fields.
- Provides a single function `save_pipeline_artifacts` for saving all outputs:
[... existing code ...]

===============================================================================
"""
#==============================================================================
# 📦 Imports
#==============================================================================
from __future__ import annotations # For type hints
import json
import pathlib
import io                 # For capturing print() output
import contextlib         # For redirecting stdout
import pandas as pd
import numpy as np        # ⭐️ ADDED: Required for metadata processing 
import math
import pickle
from types import SimpleNamespace           # ⭐️ ADDED: For type hint on 'paths'
import matplotlib.pyplot as plt # ⭐️ ADDED: For type hint on 'plt.Figure'

# --- Local ---
try:
    # This is for the full tree printer in the summary
    from VT_Module_Py_System import print_output_tree
except ImportError:
    print("🚫 [DataIO] Error: Could not import VT_Module_Py_System. Tree printer will fail.")
    # Define a dummy function to prevent crashes if module is missing
    def print_output_tree(path, no_color=False):
        print(f"   (Dummy Function: Could not import {print_output_tree} from System module)")
        print(f"   > {path.name}")
        print("   > ...")


#==============================================================================
# 📚 Data Loaders
#==============================================================================

def load_pickle(file_path: str | pathlib.Path) -> dict | None:
    """
    Loads a .pkl file safely.

    Args:
        file_path (str or Path): The full path to the .pkl file.

    Returns:
        dict: The loaded Python object (usually a dictionary).
        None: If the file is not found or fails to load.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        print(f"❌ [DataIO] Error: Pickle file not found at {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ [DataIO] Loaded pickle file: {file_path.name}")
        return data
    except Exception as e:
        print(f"❌ [DataIO] Error: Failed to load pickle file {file_path.name}. Reason: {e}")
        return None


def load_metadata_excel(parent_path: pathlib.Path, column_list: list) -> pd.DataFrame:
    """
    Finds and loads the 'Metadata.xlsx' (or .xls, .csv) file from the
    parent directory, loading *only* the specified columns.

    Args:
        parent_path (pathlib.Path): The parent folder to search in.
        column_list (list): A list of specific column names to load.

    Returns:
        pd.DataFrame: A DataFrame containing *only* the columns specified
                      in column_list.
        
    Raises:
        FileNotFoundError: If no metadata file is found.
    """
    print(f"🟠 [DataIO] Searching for Metadata file in: {parent_path.name}")
    
    # --- 1. Find the file (Excel or CSV) ---
    search_patterns = ["*.xlsx", "*.xls", "*.csv"]
    meta_file_path = None
    
    # ⭐️ MODIFIED v1.3.2: Make search more flexible.
    # Now searches for *any* file ending in .xlsx, .xls, or .csv
    # that contains "metadata" (case-insensitive) in its name.
    
    all_files = []
    for pattern in search_patterns:
        all_files.extend(parent_path.glob(pattern))
    
    for file_path in all_files:
        if "metadata" in file_path.name.lower():
            meta_file_path = file_path
            print(f"   ✅ [DataIO] Found metadata file: {meta_file_path.name}")
            break
    
    if meta_file_path is None:
        raise FileNotFoundError(f"❌ [DataIO] No metadata file (containing 'metadata' and ending in .xlsx, .xls, or .csv) found in {parent_path}")

    # --- 2. Load the file using the specified columns ---
    try:
        if meta_file_path.suffix.lower() == ".csv":
            df_meta = pd.read_csv(
                meta_file_path, 
                engine="c", 
                usecols=lambda c: c in column_list, # Load only requested
                on_bad_lines="warn"
            )
        else: # .xlsx or .xls
            df_meta = pd.read_excel(
                meta_file_path, 
                usecols=column_list, # Load only requested
                engine="openpyxl" # or "xlrd" if needed
            )
            
        # --- 3. Validate and re-order columns ---
        # Ensure DataFrame has all requested columns, even if empty
        for col in column_list:
            if col not in df_meta.columns:
                df_meta[col] = pd.NA
        
        # Return in the exact order requested
        df_meta = df_meta[column_list]
        
        print(f"   ✅ [DataIO] Successfully loaded {len(df_meta.columns)} metadata columns.")
        return df_meta

    except ValueError as e:
        # This often happens if a column in column_list is not in the file
        print(f"❌ [DataIO] Error loading metadata: {e}")
        print(f"   ❗️ Please check if all {len(column_list)} required columns are in the file.")
        # Return an empty DataFrame with the expected columns to prevent crashes
        return pd.DataFrame(columns=column_list)
    except Exception as e:
        print(f"❌ [DataIO] An unexpected error occurred loading {meta_file_path.name}: {e}")
        return pd.DataFrame(columns=column_list)


def load_metadata_snapshot(file_path: str | pathlib.Path, column_list: list) -> pd.DataFrame | None:
    """
    Loads the 'PyPD_Metadata_...' file created by Stage 1.
    This is used by Stage 2 and later.

    Args:
        file_path (str or Path): The *exact* path to the snapshot CSV.
        column_list (list): A list of *required* columns to check for.

    Returns:
        pd.DataFrame: The loaded metadata snapshot.
        None: If the file is not found or fails to load.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        print(f"❌ [DataIO] Error: Metadata snapshot not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, engine="c")
        
        # --- Validate ---
        missing_cols = [col for col in column_list if col not in df.columns]
        if missing_cols:
            print(f"❌ [DataIO] Error: Snapshot {file_path.name} is missing required columns:")
            for col in missing_cols:
                print(f"   - {col}")
            return None
            
        print(f"✅ [DataIO] Loaded metadata snapshot: {file_path.name}")
        return df
        
    except Exception as e:
        print(f"❌ [DataIO] Error: Failed to load snapshot {file_path.name}. Reason: {e}")
        return None

# =============================================================================
# CSV Formatting
# =============================================================================

def _round_sig(x: float, sig: int) -> float:
    if x == 0 or not np.isfinite(x): return x
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

VS_VOLTAGE_COLS = [
    "Eipp (V1)", "V2", "Vc peak (V3)", "Emc (V4)", "E IPend (v5)",
    "Vacc leading (V2=V1)", "ΔV (V3-V1)", "ΔEp (V3-V2)", "Vacc trailing (V4=V3)"
]

def format_metadata_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rounding/sig-figs for metadata & summaries (numeric dtypes only)."""
    out = df.copy()
    _num = lambda s: pd.to_numeric(s, errors="coerce")

    # time (0.1 µs)
    for col in [c for c in out.columns if "PhaseWidth" in c or "InterphaseDelay" in c]:
        out[col] = _num(out[col]).round(1)

    # geometry (4 s.f.)
    for col in [c for c in out.columns if "ElectrodeDiameter" in c or "GSA" in c]:
        out[col] = _num(out[col]).apply(
            lambda v: np.nan if pd.isna(v) else _round_sig(float(v), 4)
            )
        
    # current µA (4 s.f.)
    for col in [c for c in out.columns if "Current(" in c and "µA" in c]:
        out[col] = _num(out[col]).apply(
            lambda v: np.nan if pd.isna(v) else _round_sig(float(v), 4)
            )

    # charge & density (3 s.f.)
    for col in [c for c in out.columns if "ChargePerPhase" in c or "ChargeDensity" in c]:
        out[col] = _num(out[col]).apply(
            lambda v: np.nan if pd.isna(v) else _round_sig(float(v), 3)
            )

    # frequency & counts (ints)
    for col in [c for c in out.columns if c.endswith("(Hz)") or c == "TotalPulses"]:
        out[col] = _num(out[col]).round(0).astype("Int64")

    # TotalDays (4 s.f)
    for col in [c for c in out.columns if c in ["TotalDays"]]:
        out[col] = _num(out[col]).apply(
            lambda v: np.nan if pd.isna(v) else _round_sig(float(v), 4)
            )
    # TotalHours, TotalMin, TotalSec (3 s.f)
    for col in [c for c in out.columns if c in ["TotalHours", "TotalMin", "TotalSec"]]:
        out[col] = _num(out[col]).apply(
            lambda v: np.nan if pd.isna(v) else _round_sig(float(v), 3)
            )
    # TotalPulses → integer (already handled above but re-ensure)
    if "TotalPulses" in out.columns:
        out["TotalPulses"] = _num(out["TotalPulses"]).round(0).astype("Int64")

    # voltages (6 s.f.)
    for col in [c for c in VS_VOLTAGE_COLS if c in out.columns]:
        out[col] = _num(out[col]).apply(
            lambda v: np.nan if pd.isna(v) else _round_sig(float(v), 6)
            )

    return out

#==============================================================================
# 💎 METADATA PROCESSING (Moved from Stage 1)
#==============================================================================

def process_metadata_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data type corrections, calculates derived fields,
    and re-orders the final metadata snapshot DataFrame.
    
    This function modifies and returns the DataFrame passed to it.
    
    Args:
        df (pd.DataFrame): The raw, loaded metadata DataFrame.
        
    Returns:
        pd.DataFrame: The processed DataFrame with derived columns and
                      in the final specified order.
    """
    print("   ℹ️ [DataIO] Running metadata post-processing (typing, calculations, re-ordering)...")
    
    # --- Part 1: Typing & Dtypes ---
    print("      - Original dtypes:")
    print(df.dtypes)
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str)
        print("      - Forced 'Date' to string.")

    # --- Part 2: Calculations (from Template_v2) ---
    print("      - (Re)Calculating derived charge and GSA values...")
    try:
        # Convert all calculation inputs to numeric, just in case
        cols_to_convert = ['ElectrodeDiameter(cm)', 'Current(µA)', 'PhaseWidth(µs)']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # This is just a warning, not a fatal error
                print(f"      - WARNING: Calculation input column '{col}' not found. Skipping.")
        
        # Formula 1: ElectrodeGSA(cm^2)
        if 'ElectrodeGeometry' in df.columns and 'ElectrodeDiameter(cm)' in df.columns:
            df['ElectrodeGSA(cm^2)'] = np.where(
                df['ElectrodeGeometry'] == 'Circle',
                np.pi * (df['ElectrodeDiameter(cm)'] / 2)**2,
                np.nan 
            )
        
        # Formula 2-6: Charge calculations
        if 'Current(µA)' in df.columns and 'PhaseWidth(µs)' in df.columns:
            # Create the new columns
            df['ChargePerPhase(pC)'] = df['Current(µA)'] * df['PhaseWidth(µs)']
            df['ChargePerPhase(nC)'] = df['ChargePerPhase(pC)'] / 1000
            df['ChargePerPhase(mC)'] = df['ChargePerPhase(nC)'] / 1_000_000
            df['ChargePerPhase(C)']  = df['ChargePerPhase(mC)'] / 1000
            
            # 6. ChargeDensityPerPhase(mC/cm^2)
            if 'ElectrodeGSA(cm^2)' in df.columns:
                df['ChargeDensityPerPhase(mC/cm^2)'] = df['ChargePerPhase(mC)'] / df['ElectrodeGSA(cm^2)']
            else:
                df['ChargeDensityPerPhase(mC/cm^2)'] = np.nan
        
        print("      - Calculations complete.")
    
    except Exception as e:
        print(f"      - 🚫 [DataIO] Error during metadata calculations: {e}. Proceeding with partial data.")
        # We don't return here, we let it continue to re-ordering

    
    # --- Part 3: ⭐️ Apply final column order ---
    print("      - Applying final column order...")
    
    # Define the 18 loaded columns + 6 calculated columns
    FINAL_COLUMN_ORDER = [
        # Loaded (23)
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
        "TimeRecorded",	
        "TotalDays", 
        "TotalHours", 
        "TotalMin",
        "TotalSec", 
        "TotalPulses",
        # Calculated (6)
        'ElectrodeGSA(cm^2)',
        'ChargePerPhase(pC)',
        'ChargePerPhase(nC)',
        'ChargePerPhase(mC)',
        'ChargePerPhase(C)',
        'ChargeDensityPerPhase(mC/cm^2)'
    ]

    # Get all columns currently in the DataFrame
    all_current_columns = list(df.columns)
    
    # Filter the final list to only include columns that actually exist
    ordered_cols = [col for col in FINAL_COLUMN_ORDER if col in all_current_columns]
    
    # Get any "extra" columns that aren't in our main list (should be none, but just in case)
    extra_cols = [col for col in all_current_columns if col not in FINAL_COLUMN_ORDER]
    
    # Return the DataFrame with the 24 ordered columns first,
    # followed by any extras
    print("   ✅ [DataIO] Metadata post-processing complete.")
    return df[ordered_cols + extra_cols]


#==============================================================================
# 💾 Artifact Exporters
#==============================================================================

def save_pipeline_artifacts(
        paths: "SimpleNamespace", # From VT_Module_Py_System
        summary_data: dict,
        summary_base_name: str,
        figures_to_save: dict[str, "plt.Figure"] = {},
        dataframes_csv: dict[str, pd.DataFrame] = {},
        dataframes_extra: dict[str, pd.DataFrame] = {},
        pickles_stage: dict[str, object] = {},
        pickles_extra: dict[str, object] = {},
        save_formats: tuple = ("png", "svg")
    ):
    """
    Saves all pipeline artifacts (summaries, plots, CSVs, pickles)
    to their respective folders within the output structure.
    
    This function creates the JSON, TXT, and all specified files.
    """
    print(f"\n🟠 [DataIO] Saving artifacts to: {paths.stage.name}")
    
    # --- 1. Save JSON Summary ---
    json_path = paths.stage / f"{summary_base_name}.json"
    try:
        # Use a custom encoder to handle non-serializable types like Path
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, pathlib.Path):
                    return str(obj)
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                if isinstance(obj, set):
                    return list(obj)
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)
                
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=4, cls=CustomEncoder)
        print(f"   ✅ Saved JSON summary: {json_path.name}")
    except Exception as e:
        print(f"   ❌ Error saving JSON summary: {e}")

    # --- 2. Save Figures ---
    if figures_to_save and paths.png:
        for base_name, fig in figures_to_save.items():
            for fmt in save_formats:
                # Select the correct folder (png or svg)
                folder = paths.png if fmt == "png" else (paths.svg if fmt == "svg" else None)
                if folder:
                    try:
                        file_path = folder / f"{base_name}.{fmt}"
                        fig.savefig(file_path, bbox_inches="tight")
                    except Exception as e:
                        print(f"   ❌ Error saving figure {base_name}.{fmt}: {e}")
        print(f"   ✅ Saved {len(figures_to_save)} figures (formats: {save_formats})")

    # --- 3. Save DataFrames (CSVs) ---
    if dataframes_csv and paths.csv:
        for file_name, df in dataframes_csv.items():
            try:
                df.to_csv(paths.csv / file_name, index=False)
            except Exception as e:
                print(f"   ❌ Error saving CSV {file_name}: {e}")
        print(f"   ✅ Saved {len(dataframes_csv)} CSVs to {paths.csv.name}")
        
    if dataframes_extra and paths.extra:
        for file_name, df in dataframes_extra.items():
            try:
                df.to_csv(paths.extra / file_name, index=False)
            except Exception as e:
                print(f"   ❌ Error saving extra CSV {file_name}: {e}")
        print(f"   ✅ Saved {len(dataframes_extra)} extra CSVs to {paths.extra.name}")

    # --- 4. Save Pickles ---
    if pickles_stage and paths.stage:
        for file_name, data in pickles_stage.items():
            try:
                with open(paths.stage / file_name, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"   ❌ Error saving pickle {file_name}: {e}")
        print(f"   ✅ Saved {len(pickles_stage)} pickles to {paths.stage.name}")

    if pickles_extra and paths.extra:
        for file_name, data in pickles_extra.items():
            try:
                with open(paths.extra / file_name, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"   ❌ Error saving extra pickle {file_name}: {e}")
        print(f"   ✅ Saved {len(pickles_extra)} extra pickles to {paths.extra.name}")

    # --- 5. Save Detailed TXT Summary ---
    # This is written last as it includes the folder tree
    txt_path = paths.stage / f"{summary_base_name}.txt"
    try:
        _write_summary_txt(txt_path, paths, summary_data)
        print(f"   ✅ Saved TXT summary: {txt_path.name}")
    except Exception as e:
        print(f"   ❌ Error saving TXT summary: {e}")

    print("✅ [DataIO] All artifacts saved.")


def _write_summary_txt(txt_path: pathlib.Path, paths: "SimpleNamespace", summary_data: dict):
    """
    Private helper to write the detailed .txt summary file, including
    a snapshot of the generated folder tree.
    """
    
    def _format_list(item_list, indent=4):
        if not item_list: return " " * indent + "- (None)\n"
        return "".join([f"{' ' * indent}• {item}\n" for item in item_list])

    with open(txt_path, "w") as f:
        # --- 1. Header ---
        f.write("=" * 80 + "\n")
        f.write(f"VT PIPELINE SUMMARY: {summary_data.get('script', {}).get('name', 'N/A')}\n")
        f.write("=" * 80 + "\n")
        
        # --- 2. Run Info ---
        f.write(f"Timestamp:       {summary_data.get('timestamp', 'N/A')}\n")
        f.write(f"Script Version:  {summary_data.get('script', {}).get('version', 'N/A')}\n")
        f.write(f"Input Folder:    {summary_data.get('input_folder', 'N/A')}\n")
        f.write(f"Output Folder:   {summary_data.get('output_folder', 'N/A')}\n")
        f.write(f"Files Processed: {summary_data.get('n_files_processed', 'N/A')}\n")
        f.write("\n")

        # --- 3. Processed Files ---
        f.write("--- Processed Files ---\n")
        f.write(_format_list(summary_data.get('file_keys', [])))
        f.write("\n")
        
        # --- 4. Pulse Parameters ---
        params = summary_data.get('pulse_params', {})
        if params:
            f.write("--- Pulse Detection Parameters ---\n")
            for key, val in params.items():
                f.write(f"   {key:<25} = {val}\n")
            f.write("\n")

        # --- 5. Offset Summary ---
        offsets = summary_data.get('offset_summary_µs', {})
        if offsets:
            f.write("--- Pulse Time Offset Summary (µs) ---\n")
            f.write(f"   {'File':<15} | {'Status':<10} | {'Detected Start':<16} | {'Offset Start':<14} | {'Delta':<10}\n")
            f.write("   " + "-" * 70 + "\n")
            for key, val in offsets.items():
                # ⭐️ ADDED .get() and default 'N/A' to prevent crashes if a key is missing
                status = val.get('status', 'N/A')
                d_start = val.get('detected_start_µs')
                o_start = val.get('offset_start_µs')
                delta = val.get('delta_µs')
                
                # Format only if not None
                d_start_str = f"{d_start:<16.2f}" if d_start is not None else f"{'N/A':<16}"
                o_start_str = f"{o_start:<14.2f}" if o_start is not None else f"{'N/A':<14}"
                delta_str = f"{delta:<10.2f}" if delta is not None else f"{'N/A':<10}"

                f.write(f"   {key:<15} | {status:<10} | {d_start_str} | {o_start_str} | {delta_str}\n")
            f.write("\n")

        # --- 6. Generated Artifacts ---
        f.write("--- Generated Artifacts ---\n")
        artifacts = summary_data.get('artifacts', {})
        
        # Metadata
        # Read keys provided by Stage 1 (calculated) or Stage 2 (summary_csv)
        if 'metadata_calculated' in artifacts:
            # Stage 1 keys
            f.write(f"Metadata (Loaded):    {artifacts.get('metadata_loaded', 'N/A')}\n")
            f.write(f"Metadata (Calculated): {artifacts.get('metadata_calculated', 'N/A')}\n")
        elif 'summary_csv' in artifacts:
            # Stage 2 keys
            f.write(f"Metadata (Reference): {artifacts.get('metadata_csv', 'N/A')}\n")
            f.write(f"Summary CSV:          {artifacts.get('summary_csv', 'N/A')}\n")
        # (BatchPlotter has no metadata keys and will be skipped, which is correct)

        # Plots
        plots = artifacts.get("plots", [])
        if plots: f.write("Plots:\n" + _format_list(plots) + "\n")
        
        # CSVs
        csvs = artifacts.get("csvs", [])
        if csvs: f.write("CSVs:\n" + _format_list(csvs) + "\n")
        
        # Pickles
        pickles = artifacts.get("pickles", [])
        if pickles: f.write("Pickles:\n" + _format_list(pickles) + "\n")
        
        # Summary Files
        f.write("Summaries:\n")
        f.write(f"    • {txt_path.with_suffix('.json').name}\n")
        f.write(f"    • {txt_path.name}\n\n")

        # --- 7. Append Folder Tree ---
        f.write("\n🌳 Output Folder Tree\n")
        f.write("-" * 40 + "\n")
        # Create a string buffer to capture the print output
        tree_output_buffer = io.StringIO() # Renamed buffer for clarity
        try:
            # Temporarily redirect stdout to the string buffer
            with contextlib.redirect_stdout(tree_output_buffer):
                # Call the tree printer, targeting the 'stage' directory
                print_output_tree(paths.stage, no_color=True)
                
            # Write the *cleaned* output to the file
            f.write(tree_output_buffer.getvalue())
    
        except Exception as e:
            f.write(f"   🚫 [DataIO] Error: Could not generate folder tree: {e}\n")

#==============================================================================
# 📤 Module Exports
#==============================================================================
# Define what gets imported by 'from VT_Module_Py_DataIO import *'

__all__ = [
    # Data Loaders
    "load_pickle",
    "load_metadata_excel", 
    "load_metadata_snapshot",
    "format_metadata_numeric",
    
    # ⭐️ ADDED ⭐️
    "process_metadata_snapshot", 
    
    # Data Exporters
    "save_pipeline_artifacts"

]


