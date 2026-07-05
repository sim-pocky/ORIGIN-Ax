# ORIGIN-Ax

Anonymous review link:
https://anonymous.4open.science/r/ORIGIN-Ax

This is a lean release snapshot for the ORIGIN-Ax manuscript. It is kept under
100 files for convenient GitHub web upload. Raw spatial transcriptomics matrices
and large candidate-axis arrays are not included.

## Layout

- `origin_ax/`: core ORIGIN-Ax implementation and lightweight audit scripts.
- `results_v1/`: compact summary/audit CSV files used by the manuscript.
- `paper/`: IEEE LaTeX source and PDF figures.
- `paper_assets/`: compact plotting inputs.

## Minimal Checks

```bash
pip install -r requirements.txt
python origin_ax/check_final_results.py
```

Some full real-data reruns require local raw/processed spatial transcriptomics
data and detailed intermediate tables that are therefore not included in this
lightweight release. The calibration/baseline/reference audit script and its
compact output tables are included for inspection.
