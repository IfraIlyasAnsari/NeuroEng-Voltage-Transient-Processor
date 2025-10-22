#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
NeuroEng Voltage Transient Processor - Folder Paths
===============================================================================

Author:     Ifra Ilyas Ansari
Created:    2025-Oct-20
Updated:    2025-Oct-20
Code:       VT_Processor_Py_Paths.py
Version:    v1.2

DESCRIPTION
---------------------------
- Defines canonical stage names for the processing pipeline.
- Provides a function `create_output_structure` to generate a standardized
  directory tree for pipeline artifacts.
- Provides a helper `get_parent_data_folder_from_child` to find the
  root data folder from a nested file path.

===============================================================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path 
from typing import Optional

# Canonical stage names/IDs you use across scripts
STAGE_PULSE_DETECTOR   = ("PulseDetector",  "PD")  # stage_name, stage_id
STAGE_VOLTAGE_SELECTOR = ("VoltageSelector","VS")
STAGE_BATCH_PLOTTER    = ("BatchPlotter",   "BP")

@dataclass(frozen=True)
class Paths:
    """Typed container of all output folders for a single run."""
    parent: Path
    output: Path
    stage: Path
    processed: Path
    csv: Path
    png: Path
    svg: Optional[Path]     # only for BP
    extra: Optional[Path]   # only for PD
    timestamp: str

#---------------------------
# 📤 📂 CREATE OUTPUT FOLDERS
#---------------------------
def create_output_structure(
    folder_path_parent: str | os.PathLike,
    stage_name: str,      # "PulseDetector" | "VoltageSelector" | "BatchPlotter"
    stage_id: str,        # "PD" | "VS" | "BP"
    timestamp_str: Optional[str] = None,
) -> Paths:
    """
    Creates (if missing) and returns the full path structure:
    Structure:
          Parent_folder/
          └── VT_Py_Outputs/
              └── VT_Py_Outputs_{stage_name}_{YYYYMMDD_HHMM}/
                  └── Py{ID}_ProcessedData_Files/
                      ├── Py{ID}_CSV_Files/
                      ├── Py{ID}_PNG_Files/
                      ├── [Py{ID}_ExtraInfo_Files]/  (PD only)
                      └── [Py{ID}_SVG_Files]/        (BP only)

    	► folder_path_parent
    		○ folder_path_output_main
    			§ folder_path_output_stage
                        folder_path_output_processed
                            folder_path_output_processed_sub
    """
    
    folder_path_parent = Path(folder_path_parent).resolve()
    
    # timestamp_str for folder naming
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

    # 1) Define paths
    folder_path_output_main = folder_path_parent / "VT_Py_Outputs"
    folder_path_output_stage = folder_path_output_main / f"VT_Py_Outputs_{stage_name}_{timestamp_str}"
    folder_path_output_processed = folder_path_output_stage / f"Py{stage_id}_ProcessedData_Files"
    folder_path_output_processed_csv = folder_path_output_processed / f"Py{stage_id}_CSV_Files"
    folder_path_output_processed_png = folder_path_output_processed / f"Py{stage_id}_PNG_Files"
    
    # 2) Create required directories
    # .mkdir() with parents=True works like `mkdir -p`
    folder_path_output_processed_csv.mkdir(parents=True, exist_ok=True)
    folder_path_output_processed_png.mkdir(parents=True, exist_ok=True)

    # 3) Handle optional stage-specific folders
    folder_path_output_processed_extra: Optional[Path] = None
    folder_path_output_processed_svg: Optional[Path] = None

    if stage_id == "PD":
        folder_path_output_processed_extra = folder_path_output_processed / f"Py{stage_id}_ExtraInfo_Files"
        folder_path_output_processed_extra.mkdir(parents=True, exist_ok=True)

    if stage_id == "BP":
        folder_path_output_processed_svg = folder_path_output_processed / f"Py{stage_id}_SVG_Files"
        folder_path_output_processed_svg.mkdir(parents=True, exist_ok=True)
        
    return Paths(
        parent=folder_path_parent,
        output=folder_path_output_main,
        stage=folder_path_output_stage,
        processed=folder_path_output_processed,
        csv=folder_path_output_processed_csv,
        png=folder_path_output_processed_png,
        svg=folder_path_output_processed_svg,
        extra=folder_path_output_processed_extra,
        timestamp=timestamp_str,
    )
#--------------------
# 📁 METADATA PARENT FOLDER 
#--------------------
def get_parent_data_folder_from_child(path_inside_outputs: str | os.PathLike) -> str:
    """
    If `path_inside_outputs` lives under .../VT_Py_Outputs/<anything>/...,
    return the directory *above* 'VT_Py_Outputs'; else return dirname(path).

    Uses pathlib for robust path traversal.
    """
    # --- REFACTOR: Use pathlib for robust parent traversal ---
    try:
        # .resolve() handles ".." and makes path absolute
        path = Path(path_inside_outputs).resolve()
        
        # If the provided path is a file, start from its directory
        if path.is_file():
            path = path.parent

        # Iterate up the directory tree
        for p in path.parents:  # .parents -> /a/b, /a, /
            if p.name == "VT_Py_Outputs":
                # Return the parent of "VT_Py_Outputs"
                return str(p.parent)

        # Fallback: 'VT_Py_Outputs' not in path, return original dir
        return str(path)

    except (IOError, TypeError):
        # Fallback for invalid paths
        return os.path.dirname(str(path_inside_outputs))
    
__all__ = [
    "Paths",
    "create_output_structure",
    "get_parent_data_folder_from_child",
    "STAGE_PULSE_DETECTOR",
    "STAGE_VOLTAGE_SELECTOR",
    "STAGE_BATCH_PLOTTER",
]
