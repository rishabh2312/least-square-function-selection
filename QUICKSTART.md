# Ideal Function Analyzer - Quick Start Guide

Get the Ideal Function Analyzer up and running in 5 minutes!

## 📋 Prerequisites

### System Requirements
- **Python 3.11 or higher** (tested with Python 3.11.9)
- **Git** for cloning the repository
- **Operating System**: Windows, macOS, or Linux

### Python Dependencies
All dependencies are in `requirements.txt`:
- `pandas >= 2.0.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical computations
- `sqlalchemy >= 2.0.0` - Database ORM
- `bokeh >= 3.0.0` - Interactive visualizations
- `pytest >= 7.0.0` - Testing framework

## 🚀 Installation & Execution

### macOS / Linux

```bash
# 1. Clone the repository (develop branch)
git clone -b develop https://github.com/yourusername/ideal-function-analyzer.git
cd ideal-function-analyzer

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the program
python run.py
```

### Windows (Command Prompt)

```cmd
# 1. Clone the repository (develop branch)
git clone -b develop https://github.com/rishabh2312/least-square-function-selection.git
cd ideal-function-analyzer

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the program
python run.py
```

### Windows (PowerShell)

```powershell
# 1. Clone the repository (develop branch)
git clone -b develop https://github.com/yourusername/ideal-function-analyzer.git
cd ideal-function-analyzer

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the program
python run.py
```

## 📂 Important Notes

- **All development work is in the `develop` branch**
- The commands above automatically clone the develop branch with `-b develop`
- Make sure you're on the develop branch: `git branch` should show `* develop`

## ⚙️ What Happens Next?

Once you run `python run.py`, the program will:

1. ✅ Load training data, ideal functions, and test data from CSV files
2. ✅ Create a SQLite database (`ideal_functions.db`)
3. ✅ Select 4 best-fitting ideal functions using least-squares method
4. ✅ Map test points to selected ideal functions using √2 × max_deviation threshold
5. ✅ Generate 6 interactive visualizations in `output/` folder
6. ✅ Display summary statistics in console

### Expected Console Output

```
INFO | Selection complete: [
  {'training_col': 'y1', 'best_ideal_fun': 13, 'best_min_ssd': 34.08},
  {'training_col': 'y2', 'best_ideal_fun': 24, 'best_min_ssd': 33.45},
  {'training_col': 'y3', 'best_ideal_fun': 36, 'best_min_ssd': 35.57},
  {'training_col': 'y4', 'best_ideal_fun': 40, 'best_min_ssd': 34.99}
]

Total test points: 100, Assigned: 34, Unassigned: 66

ALL VISUALIZATIONS COMPLETE!
```

## View Results

Open the HTML files in your web browser:

```bash
# macOS
open output/1_complete_overview.html

# Linux
xdg-open output/1_complete_overview.html

# Windows
start output/1_complete_overview.html
```

Or simply navigate to the `output/` folder and double-click any HTML file.

## 📊 All 6 Visualizations

The program generates 6 interactive HTML visualizations in the `output/` folder:

| # | Visualization | Description |
|---|--------------|-------------|
| 1 | **Complete Overview** | All 50 ideal functions, 4 selected ideals, training data, and test points |
| 2 | **Deviation Heatmap** | Test points colored by deviation magnitude (color gradient) |
| 3 | **Training vs Ideal Grid** | 2×2 grid showing each training function vs its matched ideal |
| 4 | **Regression Analysis** | Residual plots with R², RMSE, MAE, and SSD metrics |
| 5 | **Training vs Ideal Overlay** | All 4 training-ideal pairs overlayed in a single plot |
| 6 | **Assigned vs Unassigned** | Green circles (assigned) vs red X (rejected) test points |

### Visualization Features
- 🖱️ **Interactive**: Pan, zoom, and hover for details
- 📊 **Statistical Metrics**: R², RMSE, MAE displayed
- 🎨 **Color-Coded**: Clear visual distinctions
- 💾 **Exportable**: Save as images directly from browser

## Common Issues

### "python: command not found"
➡️ Try `python3` instead of `python`

### "ModuleNotFoundError"
➡️ Make sure virtual environment is activated (you should see `(.venv)` in your terminal)
➡️ Re-run: `pip install -r requirements.txt`

### Virtual environment won't activate
➡️ **macOS/Linux**: `source .venv/bin/activate`
➡️ **Windows CMD**: `.venv\Scripts\activate.bat`
➡️ **Windows PowerShell**: `.venv\Scripts\Activate.ps1`

### Permission denied (macOS/Linux)
➡️ Run: `chmod +x run.py`

## 📁 Project Structure

```
ideal-function-analyzer/
├── data/                    # Input CSV files
│   ├── train.csv           # 4 training functions (400 points each)
│   ├── ideal.csv           # 50 ideal functions (400 points each)
│   └── test.csv            # 100 test points
├── src/                     # Source code
│   ├── database/           # SQLAlchemy models & operations
│   ├── loader/             # CSV reader
│   ├── service/            # Core algorithm (least-squares)
│   └── visualization/      # Bokeh visualization generators
├── output/                  # Generated visualizations (HTML)
├── tests/                   # Unit tests
├── run.py                   # Main entry point
├── requirements.txt         # Dependencies
└── QUICKSTART.md           # This file
```

## 🧮 Algorithm Overview

### Step 1: Ideal Function Selection
- For each of 4 training functions (Y1-Y4)
- Calculate SSD (Sum of Squared Deviations) against all 50 ideal functions
- Select the ideal with minimum SSD
- **Result**: 4 selected ideal functions (e.g., Ideals 13, 24, 36, 40)

### Step 2: Test Point Mapping
- For each of 100 test points
- Calculate deviation from each of the 4 selected ideals
- If deviation ≤ √2 × max_training_deviation → **Assign**
- Otherwise → **Reject** (unassigned)
- **Result**: ~34% assigned, ~66% unassigned (typical)

## 🧪 Running Tests

```bash
# Activate virtual environment first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run tests
pytest

# Run with detailed output
pytest -v
```

