#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 19:24:07 2025

@author: ifraa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
NeuroEng Voltage Transient Processor - Shared Utilities
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-10-21
Code:       VT_Module_Py_Utilities.py
Version:    v1.0.0

DESCRIPTION
---------------------------
- Provides common, reusable helper functions for all scripts in the
  VT processing pipeline.
- Includes functions for:
  - Auto-detecting script version from Git tags or filename.
  - Safely opening a folder/file on any OS (Windows, macOS, Linux).
  - Prompting a user for console input with a default value.
"""
import os
import re
import subprocess
import platform
from pathlib import Path

#==============================================================================
# 🧠 AUTO-DETECT SCRIPT NAME & VERSION (from file name)
#==============================================================================

def get_script_info(script_file_path: str | Path) -> tuple[str, str]:
    """
    Returns (script_name, script_version)
    Tries to read version from Git tag, else from filename.
    """
    script_path = Path(script_file_path)
    script_name = script_path.name
    
    # Try Git tag first
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], 
            cwd=script_path.parent, # Run git in the script's dir
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return script_name, version
    except Exception:
        # Fallback: extract _vX.X from filename
        match = re.search(r"_v(\d+(?:\.\d+)*)", script_name)
        version = f"v{match.group(1)}" if match else "vX.X"
        return script_name, version

#==============================================================================
# --- USER INPUT PROMPT ---
#==============================================================================
def get_user_input(prompt_text: str, default_value: str | float | int) -> str:
    """
    Prompts a user for console input, returning default if Enter is pressed.
    """
    default_str = str(default_value)
    # Handle empty string default
    default_prompt = f"'{default_str}'" if default_str else "None"
    
    user_input = input(f"{prompt_text} [Press Enter for {default_prompt}]: ").strip()
    return user_input if user_input else default_str

def get_time_limits_console(
        default_min: float, 
        default_max: float, 
        time_unit: str = "µs"
    ) -> tuple[float, float]:
    """
    Prompts user for a min and max time limit, with defaults.
    """
    print(f"\n👉 🧭 Set Time Scale ({time_unit}): ")
    try:
        xmin_str = get_user_input(f"   ✏️ 🔽 Min ({time_unit})", default_min)
        xmax_str = get_user_input(f"   ✏️ 🔼 Max ({time_unit})", default_max)
        xmin = float(xmin_str)
        xmax = float(xmax_str)
    except ValueError:
        print("🚫 Invalid input. Using default values.")
        xmin, xmax = default_min, default_max

    print(f"   ✅ Time Scale ({time_unit}): {xmin} , {xmax}")
    return xmin, xmax

#==============================================================================
# --- SAFE FILE/FOLDER OPENER ---
#==============================================================================
def safe_open(path_to_open: str | Path):
    """
    Safely opens a file or folder path on any OS.
    """
    try:
        path_str = str(path_to_open)
        # folder_to_open = str(paths.stage) in BP
        if platform.system() == "Windows":
            os.startfile(path_str)
        elif platform.system() == "Darwin":  # macOS
            os.system(f"open '{path_str}'")
        else: # Linux
            os.system(f"xdg-open '{path_str}'")
    except Exception as e:
        print(f"🚫 ERROR: Could not auto-open '{path_str}': {e}")

#==============================================================================

__all__ = [
    "get_script_info",
    "safe_open",
    "get_user_input",
    "get_time_limits_console", 
]