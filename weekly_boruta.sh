#!/bin/bash
#
# Weekly Boruta Feature Selection Pipeline
# Runs every Sunday at 1pm via cron
# Must complete before 3pm for Diamond Hands to pick up results
#
# Crontab entry:
# 0 13 * * 0 /path/to/boruta-builder/weekly_boruta.sh >> /var/log/boruta-weekly.log 2>&1
#
# Monitoring: Create a check at https://healthchecks.io and set HEALTHCHECK_URL
#
# Methodology:
# - Uses significant moves labeling (0.5% threshold) to filter noise
# - Runs Boruta on both 30-day and 90-day windows
# - Combines results: only features confirmed in BOTH windows are used
# - Importance scores are averaged across both windows
# - This finds features robust across both recent and historical regimes

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEALTHCHECK_URL="https://hc-ping.com/ee0538cb-757f-486f-ac5a-2c549f0616c7"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
OUTPUT_DIR="/var/www/diamond-hands/models"
DATE=$(date +%Y-%m-%d)

echo "========================================"
echo "Boruta Weekly Pipeline - $(date)"
echo "========================================"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate CSVs for both time windows
echo ""
echo "[Step 1/4] Generating features CSVs from PostgreSQL..."
echo "Started at: $(date)"

cd "$SCRIPT_DIR"

echo "Generating 30-day CSV (significant moves, 0.5% threshold)..."
"$VENV_PYTHON" build_features_csv.py --days 30 --significant-moves -o "boruta-features-30day.csv"

echo "Generating 90-day CSV (significant moves, 0.5% threshold)..."
"$VENV_PYTHON" build_features_csv.py --days 90 --significant-moves -o "boruta-features-90day.csv"

echo "CSVs generated"
echo "Completed at: $(date)"

# Brief pause before starting Boruta
echo "Pausing 30 seconds before Boruta analysis..."
sleep 30

# Step 2: Run Boruta on both windows
echo ""
echo "[Step 2/4] Running Boruta feature selection on both windows..."
echo "Started at: $(date)"

echo "Running 30-day Boruta..."
"$VENV_PYTHON" boruta_analysis.py boruta-features-30day.csv -q -o feature_config_30day.json

echo "Running 90-day Boruta..."
"$VENV_PYTHON" boruta_analysis.py boruta-features-90day.csv -q -o feature_config_90day.json

echo "Boruta analyses complete"
echo "Completed at: $(date)"

# Step 3: Combine results
echo ""
echo "[Step 3/4] Combining results from both windows..."
echo "Started at: $(date)"

"$VENV_PYTHON" combine_boruta_results.py \
    feature_config_30day.json \
    feature_config_90day.json \
    --labels "30-day" "90-day" \
    -o feature_config.json

if [ ! -f "$SCRIPT_DIR/feature_config.json" ]; then
    echo "ERROR: feature_config.json not generated"
    exit 1
fi

echo "Combined config generated"
echo "Completed at: $(date)"

# Brief pause before copying
echo "Pausing 10 seconds before copying results..."
sleep 10

# Step 4: Copy results to Diamond Hands
echo ""
echo "[Step 4/4] Copying feature_config.json to Diamond Hands..."
echo "Started at: $(date)"

cp "$SCRIPT_DIR/feature_config.json" "$OUTPUT_DIR/feature_config.json"

# Also keep a dated backup
cp "$SCRIPT_DIR/feature_config.json" "$OUTPUT_DIR/feature_config-$DATE.json"

echo "Copied to: $OUTPUT_DIR/feature_config.json"
echo "Backup at: $OUTPUT_DIR/feature_config-$DATE.json"

# Cleanup old CSV files (keep last 4 weeks)
echo ""
echo "Cleaning up old CSV files..."
find "$SCRIPT_DIR" -name "boruta-features-*.csv" -mtime +28 -delete

echo ""
echo "========================================"
echo "Pipeline complete at: $(date)"
echo "========================================"

# Ping healthchecks.io on success
if [ -n "$HEALTHCHECK_URL" ]; then
    curl -fsS -m 10 --retry 5 "$HEALTHCHECK_URL" > /dev/null
    echo "Healthcheck ping sent"
fi
