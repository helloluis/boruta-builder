# Boruta Feature Selection Tool

Analyzes CSV data to identify important features using the Boruta algorithm. Outputs `feature_config.json` with confirmed, tentative, and rejected features.

## Setup (one-time)

**Requires Python 3.12** (3.11 also works). Python 3.13+ is not compatible with the Boruta package.

```bash
# Create virtual environment with Python 3.12
py -3.12 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Usage

```bash
python boruta_analysis.py your_data.csv --target target_column
```

## Common Examples

```bash
# Classification (default)
python boruta_analysis.py data.csv --target label

# Regression
python boruta_analysis.py data.csv --target price --task regression

# More iterations for better accuracy (slower)
python boruta_analysis.py data.csv --target label --max-iter 200

# Quiet mode (less output)
python boruta_analysis.py data.csv --target label -q

# Custom output file
python boruta_analysis.py data.csv --target label -o my_results.json
```

## Options

| Flag | What it does |
|------|--------------|
| `--target, -t` | Target column name (required) |
| `--task` | `classification` or `regression` |
| `--max-iter` | More = better results but slower (default: 100) |
| `--output, -o` | Output filename (default: feature_config.json) |
| `-q` | Quiet mode |

## Output

Creates `feature_config.json`:
- **confirmed_features** - definitely useful
- **tentative_features** - maybe useful, review these
- **rejected_features** - not useful
- **enabled_features** - confirmed + tentative combined

## Test Run

```bash
python boruta_analysis.py sample_data.csv --target target
```
