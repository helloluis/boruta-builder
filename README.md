# Boruta Feature Selection Tool

Analyzes CSV data to identify important features using the Boruta algorithm with SHAP-based importance ranking. Outputs `feature_config.json` with confirmed, tentative, and rejected features ranked by importance.

## Setup (one-time)

**Requires Python 3.12** (3.11 also works). Python 3.13+ is not compatible with the Boruta package.

```bash
# Windows
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Build features CSV from database

```bash
python build_features_csv.py
```

This pulls data from the Neon database (configured in `.env.local`) and creates `boruta-features-YYYY-MM-DD.csv`.

### 2. Run Boruta analysis

```bash
python boruta_analysis.py boruta-features-2025-12-16.csv
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
| `--target, -t` | Target column name (default: `target`) |
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

## Database Script

The `build_features_csv.py` script fetches data from multiple tables and merges them:

- **historical_klines** - OHLCV data with technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **fear_greed_index** - Market sentiment index
- **funding_rates** - Perpetual futures funding rates
- **news_sentiment** - News sentiment scores
- **BTC price change** - Calculated from BTCUSDT close prices

The target variable is binary: `1` if next period's close > current close, `0` otherwise.

### Environment Setup

Create `.env.local` with your Neon database URL:

```
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
```

## Weekly Automation

The `weekly_boruta.sh` script runs every **Sunday at 1pm GMT+8** via cron and:

1. Generates 90-day CSV from local PostgreSQL
2. Runs Boruta analysis with SHAP ranking
3. Copies `feature_config.json` to `/var/www/diamond-hands/models/`
4. Pings [healthchecks.io](https://healthchecks.io) on success

The Diamond Hands app picks up the updated feature config at 3pm to retrain its models.

### Crontab Setup

```bash
crontab -e
# Add: 0 5 * * 0 /path/to/boruta-builder/weekly_boruta.sh >> /var/log/boruta-weekly.log 2>&1
```

(0 5 UTC = 1pm GMT+8)

## Test Run

```bash
# Using sample data
python boruta_analysis.py sample_data.csv

# Using database data
python build_features_csv.py
python boruta_analysis.py boruta-features-2025-12-16.csv -q
```
