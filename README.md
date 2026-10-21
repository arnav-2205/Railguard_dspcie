# RailGuard

FFT-based railway track vibration anomaly screening (DSP course prototype) on the Rail-VIVID dataset.

* Start with **decisions.md** (chronological reasoning, evidence, caveats) then **system_architecture.md** (final system, DSP formulas, how to run).
* Quick start: `pip install -r requirements.txt`, `pytest -s tests`, `python scripts/download_data.py`, `python scripts/extract_features.py`, `python scripts/fit_and_evaluate.py`, `streamlit run app.py`.
* The 3 validation runs listed in `railguard/config.py` are never loaded by the code (guard in `railguard/data.py`).
