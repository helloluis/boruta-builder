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

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEALTHCHECK_URL="https://hc-ping.com/ee0538cb-757f-486f-ac5a-2c549f0616c7"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
OUTPUT_DIR="/var/www/diamond-hands/models"
DATE=$(date +%Y-%m-%d)
CSV_FILE="$SCRIPT_DIR/boruta-features-$DATE.csv"

echo "========================================"
echo "Boruta Weekly Pipeline - $(date)"
echo "========================================"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate CSV from local PostgreSQL
echo ""
echo "[Step 1/3] Generating features CSV from PostgreSQL..."
echo "Started at: $(date)"

cd "$SCRIPT_DIR"
"$VENV_PYTHON" build_features_csv.py

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file not generated: $CSV_FILE"
    exit 1
fi

echo "CSV generated: $CSV_FILE ($(du -h "$CSV_FILE" | cut -f1))"
echo "Completed at: $(date)"

# Brief pause before starting Boruta
echo "Pausing 30 seconds before Boruta analysis..."
sleep 30

# Step 2: Run Boruta analysis
echo ""
echo "[Step 2/3] Running Boruta feature selection..."
echo "Started at: $(date)"

"$VENV_PYTHON" boruta_analysis.py "$CSV_FILE" -q

if [ ! -f "$SCRIPT_DIR/feature_config.json" ]; then
    echo "ERROR: feature_config.json not generated"
    exit 1
fi

echo "Boruta analysis complete"
echo "Completed at: $(date)"

# Brief pause before copying
echo "Pausing 10 seconds before copying results..."
sleep 10

# Step 3: Copy results to Diamond Hands
echo ""
echo "[Step 3/3] Copying feature_config.json to Diamond Hands..."
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
