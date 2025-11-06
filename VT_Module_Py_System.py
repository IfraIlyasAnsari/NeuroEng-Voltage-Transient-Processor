"""
===============================================================================
NeuroEng Voltage Transient Processor - System Interactions
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-Oct-22
Modified:   2025-Oct-30
Code:       VT_Module_Py_System.py
Version:    v1.0.3

DESCRIPTION
---------------------------
- Centralizes functions related to system interactions for the VT pipeline.
- Provides GUI file pickers for selecting CSVs and PKL files.
- Includes functions for creating standardized output folder structures (`Paths`).
- Provides helpers for finding the root data folder.
- Includes functions for safely opening files/folders on any OS.
- Provides functions for printing directory trees (current or latest run).
- Includes function for getting script name and version (from filename or git).

NOTE
---------------------------
- Data I/O functions (reading/writing CSV, PKL, PNG, JSON, and TXT summaries) 
  are now in VT_Module_Py_DataIO.py.
- User input prompts are in VT_Module_Py_UserInput.py.
===============================================================================
"""

#==============================================================================
# 📦 Imports
#==============================================================================
from __future__ import annotations # Must be first non-comment line
import os
import sys
import re
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import List, Tuple, Optional

# --- GUI Imports ---
try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TKINTER = True
except ImportError:
    _HAS_TKINTER = False
    print("🚫 [System] Error: tkinter module not found. GUI file pickers will be disabled.")

# --- Git Import (Optional) ---
try:
    import git
    _HAS_GIT = True
except ImportError:
    _HAS_GIT = False


#==============================================================================
# 🧠 Script Info
#==============================================================================

def get_script_info(file_dunder: str) -> tuple[str, str]:
    """
    Gets the script name (without .py) and version (from filename or git tag).
    
    Args:
        file_dunder (str): Pass `__file__` from the calling script.
        
    Returns:
        tuple[str, str]: (script_name, script_version)
    """
    
    script_path = Path(file_dunder).resolve()
    script_name_py = script_path.name
    script_name_stem = script_path.stem # Name without .py

    # --- 1. Try to get version from filename (e.g., "script_name_v1.2.3.py") ---
    # This regex looks for "_v" followed by digits.digits.digits
    match = re.search(r"_(v[\d\.]+\d+)$", script_name_stem, re.IGNORECASE)
    if match:
        script_version = match.group(1) # e.g., "v1.2.3"
        # Clean up the name if it follows the pattern
        script_name_base = script_name_stem.replace(f"_{script_version}", "")
        return script_name_base, script_version

    # --- 2. If no version in filename, try to get from git tag ---
    if _HAS_GIT:
        try:
            repo = git.Repo(script_path, search_parent_directories=True)
            # Get the most recent tag on the current commit
            git_tag = repo.git.describe('--tags', '--abbrev=0', '--always')
            if git_tag:
                return script_name_stem, f"git:{git_tag}"
        except (git.InvalidGitRepositoryError, git.GitCommandError):
            pass # Not a git repo or no tags

    # --- 3. Fallback ---
    return script_name_stem, "v_unknown"


#==============================================================================
# 📂 Path & Folder Management
#==============================================================================

# Define standard stage names (used for folder creation and tree view)
STAGE_PULSE_DETECTOR   = ("PulseDetector",   "PyPD")
STAGE_VOLTAGE_SELECTOR = ("VoltageSelector", "PyVS")
STAGE_BATCH_PLOTTER    = ("BatchPlotter",           "PyBP")

class Paths(SimpleNamespace):
    """
    A SimpleNamespace data class for storing the standardized folder structure.
    
    Attributes:
        timestamp (str): The timestamp string (e.g., "20251030_143005").
        parent (Path): The root data folder where the input CSVs are.
        output (Path): The main 'VT_Py_Outputs' folder.
        stage (Path): The specific, timestamped run folder
        processed (Path): The 'Processed_Data' subfolder.
        csv (Path): Subfolder for primary CSVs (e.g., '.../PyPD_CSV_Files').
        png (Path): Subfolder for primary PNGs (e.g., '.../PyPD_PNG_Files').
        svg (Path): Subfolder for optional SVGs (e.g., '.../PyPD_SVG_Files').
        extra (Path): Subfolder for intermediate data (e.g., '.../PyPD_ExtraInfo_Files').
    """
    pass

def create_output_structure(
    parent_data_folder: str | Path, 
    stage_name: str,
    stage_id: str,
    create_svg_folder: bool = False,
    create_extra_folder: bool = True
) -> "Paths":
    """
    Creates a standardized, timestamped output folder structure.

    Args:
        parent_data_folder (str or Path): The path to the root data folder
                                          (where the input CSVs are).
        stage_name (str): The descriptive name for the stage (e.g., "PulseDetector").
        stage_id (str): The short ID for the stage (e.g., "PyPD").
        create_svg_folder (bool): Whether to create an SVG subfolder.
        create_extra_folder (bool): Whether to create an "ExtraInfo" subfolder.

    Returns:
        Paths (SimpleNamespace): An object containing Path objects for all
                                 created directories.
    """
    
    parent_path = Path(parent_data_folder).resolve()
    
    # 1. Define main output folder ('VT_Py_Outputs')
    # This folder sits *inside* the parent_data_folder
    output_main = parent_path / "VT_Py_Outputs"
    
    # 2. Define timestamped run folder (e.g., "VT_Py_Outputs_PulseDetector_20251030_1443")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    stage_run_folder_name = f"VT_Py_Outputs_{stage_name}_{timestamp}" 
    stage_run_path = output_main / stage_run_folder_name
    
    # 3. Define 'Processed' subfolder
    processed_path = stage_run_path / "ProcessedDataFiles"

    # 4. Define artifact subfolders (e.g., "PyPD_CSV_Files")
    csv_path   = processed_path / f"{stage_id}_CSV_Files"
    png_path   = processed_path / f"{stage_id}_PNG_Files"
    svg_path   = processed_path / f"{stage_id}_SVG_Files" if create_svg_folder else None
    extra_path = processed_path / f"{stage_id}_ExtraInfo_Files" if create_extra_folder else None

    # 5. Create all directories
    # (Create in reverse order to ensure parents exist)
    folders_to_create = [csv_path, png_path, svg_path, extra_path]
    for folder in folders_to_create:
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
            
    # 6. Store in the Paths object
    p = Paths()
    p.timestamp = timestamp
    p.parent = parent_path
    p.output = output_main
    p.stage = stage_run_path
    p.processed = processed_path
    p.csv = csv_path
    p.png = png_path
    p.svg = svg_path
    p.extra = extra_path
    
    return p 

def get_parent_data_folder_from_child(child_path: str | Path) -> Path:
    """
    Return the *experiment root* folder (the one that directly contains
    'VT_Py_Outputs' and your raw TEK files/metadata), regardless of whether
    'child_path' points to:
      - a file under a Stage folder (…/VT_Py_Outputs/VT_Py_Outputs_*/*)
      - a Stage folder itself
      - the VT_Py_Outputs folder
      - the experiment root already
    """
    p = Path(child_path).resolve()
    if p.is_file():
        p = p.parent  # go to its folder

    # If we are in a stage folder (name starts with VT_Py_Outputs_)
    if p.name.startswith("VT_Py_Outputs_"):
        return p.parent.parent  # Stage → VT_Py_Outputs → EXPERIMENT ROOT

    # If we are at VT_Py_Outputs level
    if p.name == "VT_Py_Outputs":
        return p.parent  # EXPERIMENT ROOT

    # Otherwise assume we're already at the experiment root
    return p

#==============================================================================
# 🖥️ GUI File Pickers
#==============================================================================

def _init_tk_root():
    """Initializes a hidden tkinter root window."""
    if not _HAS_TKINTER:
        return None
    root = tk.Tk()
    root.withdraw() # Hide the main window
    # Move the file dialog to the front
    root.attributes("-topmost", True)
    return root

def select_csv_files_gui() -> List[Path]:
    """
    Opens a GUI file picker to select one or more .csv files.
    
    Returns:
        List[Path]: A list of Path objects for the selected files.
                    Returns an empty list if canceled.
    """
    print("🟠 [System] Opening GUI to select CSV file(s)...")
    root = _init_tk_root()
    if root is None:
        raise ImportError("GUI selection failed: tkinter is not installed.")
        
    file_paths_str = filedialog.askopenfilenames(
        title="Select Tektronix CSV file(s)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not file_paths_str:
        print("🚫 [System] Error: User canceled selection.")
        return []
        
    file_paths_list = [Path(p) for p in file_paths_str]
    print(f"✅ [System] Selected {len(file_paths_list)} file(s).")
    return file_paths_list

def select_pickle_file_gui() -> Path | None:
    """
    Opens a GUI file picker to select a single .pkl file.
    
    Returns:
        Path | None: A Path object for the selected file, or None if canceled.
    """
    print("🟠 [System] Opening GUI to select a Stage 1 .pkl file...")
    root = _init_tk_root()
    if root is None:
        raise ImportError("GUI selection failed: tkinter is not installed.")
        
    file_path_str = filedialog.askopenfilename(
        title="Select the Stage 1 .pkl dictionary file",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not file_path_str:
        print("🚫 [System] Error: User canceled selection.")
        return None
        
    file_path = Path(file_path_str)
    print(f"✅ [System] Selected file: {file_path.name}")
    return file_path
    
#==============================================================================
# 🚀 File/Folder Operations
#==============================================================================

def safe_open(path: str | Path):
    """
    Safely opens a file or folder using the default OS application.
    (e.g., Finder on macOS, Explorer on Windows).
    
    Args:
        path (str or Path): The path to the file or folder to open.
    """
    path_str = str(Path(path).resolve())
    try:
        if platform.system() == "Windows":
            os.startfile(path_str)
        elif platform.system() == "Darwin": # macOS
            subprocess.run(["open", path_str], check=True)
        else: # Linux
            subprocess.run(["xdg-open", path_str], check=True)
    except Exception as e:
        print(f"🚫 [System] Error: Could not auto-open path '{path_str}': {e}")
        print("🚫 [System] Please open it manually.")


#==============================================================================
# 🌳 Directory Tree Printers
#==============================================================================
# (Inspired by 'tree' command)

# --- Define ANSI colors for the tree ---
class Colors:
    DIR = '\033[94m'      # Blue
    FILE = '\033[92m'     # Green
    BRANCH = '\033[90m'   # Dark Grey
    END = '\033[0m'       # Reset
    
def _get_tree_colors(no_color: bool):
    """Returns the color class or a dummy class if disabled."""
    if no_color or platform.system() == "Windows":
        class DummyColors:
            DIR = FILE = BRANCH = END = ''
        return DummyColors
    return Colors

def print_output_tree(start_path: Path, max_depth: int = 5, no_color: bool = False):
    """
    Prints a visual directory tree structure.
    
    Args:
        start_path (Path): The root directory to start the tree from.
        max_depth (int): The maximum number of levels to descend.
        no_color (bool): If True, disables ANSI color codes.
    """
    C = _get_tree_colors(no_color)
    start_path = Path(start_path).resolve()
    
    if not start_path.is_dir():
        print(f"🚫 [System] Error: Cannot print tree: '{start_path}' is not a valid directory.")
        return

    print(f"{C.DIR}📂 {start_path.name}{C.END}")
    
    _print_tree_recursive(start_path, "", max_depth, C)

def _print_tree_recursive(directory: Path, prefix: str, max_depth: int, C: Colors):
    """Recursive helper function for print_output_tree."""
    
    if max_depth <= 0:
        if any(directory.iterdir()):
            print(f"{prefix}{C.BRANCH}├── ... (max depth reached){C.END}")
        return

    # Get all items, filter out system files (like .DS_Store)
    items = [item for item in directory.iterdir() if not item.name.startswith('.')]
    # Sort files and directories separately, then combine
    dirs = sorted([d for d in items if d.is_dir()], key=lambda x: x.name.lower())
    files = sorted([f for f in items if f.is_file()], key=lambda x: x.name.lower())
    all_items = dirs + files
    
    for i, item in enumerate(all_items):
        is_last = (i == len(all_items) - 1)
        
        # Define connectors
        connector = "└── " if is_last else "├── "
        line_prefix = f"{prefix}{C.BRANCH}{connector}{C.END}"
        
        # Define the prefix for the *next* level
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"

        if item.is_dir():
            print(f"{line_prefix}{C.DIR}📂 {item.name}{C.END}")
            # Recurse
            _print_tree_recursive(item, child_prefix, max_depth - 1, C)
        else:
            print(f"{line_prefix}{C.FILE}📄 {item.name}{C.END}")

# --- Latest Run Finder ---
def _find_latest_stage_run(output_main_folder: Path, stage_name: Optional[str] = None) -> Path | None:
    """
    Finds the most recently created run folder within the main VT_Py_Outputs directory.
    """
    if not output_main_folder.exists():
        return None
        
    # Pattern: "VT_Py_Outputs_STAGE_YYYYMMDD_HHMM"
    pattern = re.compile(r"^VT_Py_Outputs_(.*)_(\d{8}_\d{4})$")
    
    latest_time = 0
    latest_folder = None
    
    for item in output_main_folder.iterdir():
        if not item.is_dir():
            continue
            
        match = pattern.match(item.name)
        if not match:
            continue # Not a timestamped run folder
            
        # Optional: Filter by stage name if provided
        if stage_name is not None:
        # Check the captured stage name from group 1
           found_stage_name = match.group(1)
           if found_stage_name != stage_name:
                continue
                
        # Parse timestamp
        try:
            # Get timestamp from group 2
            timestamp_str = match.group(2) # e.g., "20251030_1443"
            timestamp_int = int(timestamp_str) # e.g., 202510301443
            
            if timestamp_int > latest_time:
                latest_time = timestamp_int
                latest_folder = item
        except (ValueError, TypeError):
            continue # Failed to parse timestamp
            
    return latest_folder

def print_latest_output_tree(output_main_folder: Path, stage_name: Optional[str] = None, max_depth: int = 5):
    """
    Finds the most recent run folder within the main VT_Py_Outputs directory 
    (optionally filtering by stage name) and prints its tree structure.
    """
    latest_run_folder = _find_latest_stage_run(output_main_folder, stage_name=stage_name)
    
    if latest_run_folder is None:
        if stage_name:
            print(f"\n🚫 [System] Error: No timestamped run folders found under '{output_main_folder.name}' for stage '{stage_name}'.")
        else:
            print(f"\n🚫 [System] Error: No timestamped run folders found under '{output_main_folder.name}'.")
        return

    print("\n🌳 Displaying tree for the latest run found:")
    # Call the main tree printer function with the located folder
    print_output_tree(latest_run_folder, max_depth=max_depth)

#==============================================================================
# 📤 Module Exports
#==============================================================================
# Define what gets imported by 'from VT_Module_Py_System import *'

__all__ = [
    # Script Info
    "get_script_info",
    # File/Folder Operations
    "safe_open",
    # Path Management
    "Paths", 
    "create_output_structure",
    "get_parent_data_folder_from_child",
    # Stage Constants
    "STAGE_PULSE_DETECTOR",
    "STAGE_VOLTAGE_SELECTOR",
    "STAGE_BATCH_PLOTTER",
    # GUI Selectors
    "select_csv_files_gui",
    "select_pickle_file_gui", # ⭐️ ADDED v1.0.3 to fix ImportError
    # Tree Printers
    "print_output_tree",
    "print_latest_output_tree"
]
