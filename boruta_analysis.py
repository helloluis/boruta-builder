#!/usr/bin/env python3
"""
Boruta Feature Selection Analysis Tool

Accepts a CSV file with features and a target column, runs Boruta analysis,
and outputs feature_config.json with confirmed, tentative, and rejected features.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import shap
from boruta import BorutaPy
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
    return df


def run_boruta_analysis(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
    max_iter: int = 100,
    random_state: int = 42,
    n_estimators: int = 100,
    verbose: int = 1,
    warnings_list: list = None
) -> dict:
    """
    Run Boruta feature selection analysis.

    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        task_type: 'classification' or 'regression'
        max_iter: Maximum iterations for Boruta
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in the random forest
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        warnings_list: Optional list to collect warnings for logging

    Returns:
        Dictionary with feature categorization results
    """
    if warnings_list is None:
        warnings_list = []

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {list(df.columns)}")

    # Separate features and target (only numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_cols if col != target_column]

    # Warn about excluded non-numeric columns
    excluded_cols = [col for col in df.columns if col not in numeric_cols and col != target_column]
    if excluded_cols:
        print(f"Note: Excluding non-numeric columns: {excluded_cols}")

    y = df[target_column].values

    # Drop rows with missing target values first
    if pd.isna(y).any():
        missing_count = pd.isna(y).sum()
        warning_msg = f"Dropping {missing_count} rows with missing target values."
        print(f"Warning: {warning_msg}")
        warnings_list.append(warning_msg)
        mask = ~pd.isna(y)
        df = df[mask]
        y = y[mask]

    # Drop columns that are entirely NaN
    all_nan_cols = [col for col in feature_columns if df[col].isna().all()]
    if all_nan_cols:
        warning_msg = f"Dropping columns with all NaN values: {all_nan_cols}"
        print(f"Warning: {warning_msg}")
        warnings_list.append(warning_msg)
        feature_columns = [col for col in feature_columns if col not in all_nan_cols]

    # Now get features and handle missing values
    X = df[feature_columns].values
    if np.isnan(X).any():
        warning_msg = "Found NaN values in features. Filling with column means."
        print(f"Warning: {warning_msg}")
        warnings_list.append(warning_msg)
        df_features = df[feature_columns].fillna(df[feature_columns].mean())
        X = df_features.values

    print(f"\nRunning Boruta analysis...")
    print(f"  Task type: {task_type}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Samples: {len(y)}")
    print(f"  Max iterations: {max_iter}")
    print(f"  This may take 10-30+ minutes on limited hardware...")
    sys.stdout.flush()

    # Select appropriate estimator
    if task_type == "classification":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
            max_depth=5
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
            max_depth=5
        )

    # Run Boruta
    boruta = BorutaPy(
        estimator=estimator,
        n_estimators='auto',
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose
    )

    boruta.fit(X, y)
    print("  Boruta fitting complete.")
    sys.stdout.flush()

    # Categorize features
    confirmed_features = []
    tentative_features = []
    rejected_features = []

    for i, col in enumerate(feature_columns):
        if boruta.support_[i]:
            confirmed_features.append(col)
        elif boruta.support_weak_[i]:
            tentative_features.append(col)
        else:
            rejected_features.append(col)

    print(f"\nResults:")
    print(f"  Confirmed features: {len(confirmed_features)}")
    print(f"  Tentative features: {len(tentative_features)}")
    print(f"  Rejected features: {len(rejected_features)}")

    # Calculate SHAP importance scores for confirmed features
    feature_importance = {}
    if confirmed_features or tentative_features:
        print(f"\nCalculating SHAP importance scores...")
        sys.stdout.flush()

        # Get indices of confirmed + tentative features
        selected_features = confirmed_features + tentative_features
        selected_indices = [feature_columns.index(f) for f in selected_features]
        X_selected = X[:, selected_indices]

        # Fit a fresh Random Forest on selected features for SHAP
        if task_type == "classification":
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=random_state,
                max_depth=7
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=random_state,
                max_depth=7
            )
        rf.fit(X_selected, y)

        # Use a sample for SHAP if dataset is large
        if len(X_selected) > 1000:
            sample_idx = np.random.RandomState(random_state).choice(
                len(X_selected), size=1000, replace=False
            )
            X_sample = X_selected[sample_idx]
        else:
            X_sample = X_selected

        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            # Shape is (samples, features, classes) or (classes, samples, features)
            # Take mean absolute value across classes
            if shap_values.shape[0] == len(X_sample):
                # (samples, features, classes)
                mean_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                # (classes, samples, features)
                mean_shap = np.abs(shap_values).mean(axis=(0, 1))
        elif isinstance(shap_values, list):
            # Old format: list of arrays per class
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            mean_shap = shap_values.mean(axis=0)
        else:
            # 2D array (samples, features)
            mean_shap = np.abs(shap_values).mean(axis=0)

        # Create importance dict and normalize to percentages
        total_importance = mean_shap.sum()
        for i, feat in enumerate(selected_features):
            feature_importance[feat] = round(float(mean_shap[i] / total_importance * 100), 2)

        # Sort features by importance
        confirmed_features.sort(key=lambda x: feature_importance.get(x, 0), reverse=True)
        tentative_features.sort(key=lambda x: feature_importance.get(x, 0), reverse=True)

    # Enabled features = confirmed + tentative (sorted by importance)
    enabled_features = confirmed_features + tentative_features

    if confirmed_features:
        print(f"\nFeature importance (SHAP-based):")
        for col in confirmed_features:
            print(f"  {feature_importance[col]:5.2f}%  {col}")

    return {
        "enabled_features": enabled_features,
        "confirmed_features": confirmed_features,
        "tentative_features": tentative_features,
        "rejected_features": rejected_features,
        "feature_importance": feature_importance,
        "generated_at": datetime.now().isoformat(),
        "notes": "Features ranked by SHAP importance (percentage of total contribution)."
    }


def save_feature_config(results: dict, output_path: str = "feature_config.json"):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFeature config saved to: {output_path}")


def get_db_connection():
    """Get database connection for logging."""
    env_path = Path(__file__).parent / '.env.local'

    if env_path.exists():
        load_dotenv(env_path)
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            return psycopg2.connect(database_url)

    # Fall back to localhost PostgreSQL
    return psycopg2.connect(
        host='localhost',
        database='earnest_db',
        user='earnest',
        password='earnest_secure_2024',
        port=5432
    )


def log_to_database(results: dict, warnings: list, errors: list):
    """Log Boruta results to ml_training_logs table."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Build the results JSONB
        log_data = {
            "confirmed_features": results.get("confirmed_features", []),
            "tentative_features": results.get("tentative_features", []),
            "rejected_features": results.get("rejected_features", []),
            "feature_importance": results.get("feature_importance", {}),
            "total_features_analyzed": (
                len(results.get("confirmed_features", [])) +
                len(results.get("tentative_features", [])) +
                len(results.get("rejected_features", []))
            ),
            "confirmed_count": len(results.get("confirmed_features", [])),
            "tentative_count": len(results.get("tentative_features", [])),
            "rejected_count": len(results.get("rejected_features", [])),
            "generated_at": results.get("generated_at"),
        }

        # Build warnings/errors JSONB (null if empty)
        issues_data = None
        if warnings or errors:
            issues_data = json.dumps({
                "warnings": warnings if warnings else [],
                "errors": errors if errors else []
            })

        cur.execute("""
            INSERT INTO ml_training_logs (helper_name, output, errors, created_at)
            VALUES (%s, %s, %s, NOW())
        """, ("Boruta", json.dumps(log_data), issues_data))

        conn.commit()
        cur.close()
        conn.close()
        print("Results logged to ml_training_logs")

    except Exception as e:
        print(f"Warning: Could not log to database: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run Boruta feature selection analysis on a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python boruta_analysis.py data.csv --target target_column
  python boruta_analysis.py data.csv --target label --task regression
  python boruta_analysis.py data.csv --target y --max-iter 200 --output results.json
        """
    )

    parser.add_argument(
        "csv_file",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--target", "-t",
        default="target",
        help="Name of the target column in the CSV (default: target)"
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Type of ML task (default: classification)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations for Boruta (default: 100)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in random forest (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        default="feature_config.json",
        help="Output JSON file path (default: feature_config.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress Boruta progress output"
    )

    args = parser.parse_args()

    warnings_list = []
    errors_list = []

    try:
        # Load data
        df = load_csv(args.csv_file)

        # Run analysis
        results = run_boruta_analysis(
            df=df,
            target_column=args.target,
            task_type=args.task,
            max_iter=args.max_iter,
            n_estimators=args.n_estimators,
            random_state=args.seed,
            verbose=0 if args.quiet else 2,
            warnings_list=warnings_list
        )

        # Save results
        save_feature_config(results, args.output)

        # Log to database
        log_to_database(results, warnings_list, errors_list)

        print("\nDone!")
        return 0

    except Exception as e:
        error_msg = str(e)
        errors_list.append(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)

        # Still log the error to database
        log_to_database({}, warnings_list, errors_list)

        return 1


if __name__ == "__main__":
    sys.exit(main())
