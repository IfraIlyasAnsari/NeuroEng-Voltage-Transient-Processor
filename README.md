Author:     Ifra Ilyas Ansari
Created:    2025-10-15
Updated:    2025-10-21

# NeuroEng-Voltage-Transient-Processor - README
This project is a 3-stage Python pipeline for processing, analyzing, and plotting voltage transient (VT) data from Tektronix oscilloscopes.

## The Pipeline
The workflow is broken into three scripts, designed to be run in order:

1.  **`VT_Processor_PyStage1_PulseDetector.py`**
    * **Input:**  Raw `.CSV` files from the oscilloscope.
    * **Action:** Auto-detects the biphasic pulse, trims the data, and aligns all traces to a common `t=0` based on the pulse onset.
    * **Output:** A `.pkl` file containing a dictionary of cleaned, aligned DataFrames.

2.  **`VT_Processor_PyStage2_VoltageSelector.py`**
    * **Input:**  The `.pkl` file from Stage 1.
    * **Action:** Prompts the user to enter specific time points (e.g., `t1`, `t2`...) to extract key voltage values.
    * **Output:** A final summary `.csv` merging metadata with the extracted voltages, plus individual plots and a new `.pkl` file.

3.  **`VT_Processor_Py_BatchPlotter.py`**
    * **Input:**  Raw `.CSV` files.
    * **Action:** A simpler tool to quickly batch-plot multiple CSVs on one graph with custom legends and user-defined time offsets.
    * **Output:** Publication-quality plots in `PNG` and `SVG` format.

## ⚙️ Configuration
All scripts import their settings from the `VT_Module_...` files. The most important one is **`VT_Module_Py_GlobalSettings.py`**.
If your data is not loading or your plots look wrong, you probably need to edit this file:

* **To swap Voltage/Current channels:** Change `DICT_TEK_CHANNEL_MAP`.
    ```python
    # e.g., if Voltage is on CH2 and Current is on CH1
    DICT_TEK_CHANNEL_MAP = {
        "VOLTAGE": "CH2",
        "CURRENT": "CH1",
    }
    ```
* **To change plot appearance:** Edit the `rcParams.update({...})` dictionary to change fonts, line widths, and figure sizes.

## Requirements
The pipeline requires standard scientific Python libraries:
* pandas
* numpy
* matplotlib