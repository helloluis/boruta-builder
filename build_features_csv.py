#!/usr/bin/env python3
"""
Build features CSV from Neon database for Boruta analysis.

Pulls data from historical_klines, fear_greed_index, funding_rates,
and news_sentiment tables, then creates a target variable based on
whether the price goes up in the next period.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm


def get_connection():
    """Get database connection from .env.local or fall back to localhost."""
    env_path = Path(__file__).parent / '.env.local'

    if env_path.exists():
        load_dotenv(env_path)
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            print("Using DATABASE_URL from .env.local")
            return psycopg2.connect(database_url)

    # Fall back to localhost PostgreSQL
    print("No .env.local found, using localhost PostgreSQL")
    return psycopg2.connect(
        host='localhost',
        database='earnest_db',
        user='earnest',
        password='earnest_secure_2024',
        port=5432
    )


def fetch_historical_klines(conn, days: int = 90) -> pd.DataFrame:
    """Fetch historical klines with technical indicators (last N days)."""
    query = f"""
        SELECT
            symbol,
            open_time as timestamp,
            open_price as open,
            high_price as high,
            low_price as low,
            close_price as close,
            volume,
            rsi_14,
            rsi_30,
            macd,
            macd_signal,
            bb_upper,
            bb_lower,
            bb_width,
            momentum_10,
            stoch_k,
            stoch_d,
            volume_sma
        FROM historical_klines
        WHERE open_time >= NOW() - INTERVAL '{days} days'
        ORDER BY symbol, open_time
    """
    return pd.read_sql(query, conn)


def fetch_fear_greed(conn, days: int = 90) -> pd.DataFrame:
    """Fetch fear & greed index data (last N days)."""
    query = f"""
        SELECT
            index_timestamp as timestamp,
            value as fear_greed
        FROM fear_greed_index
        WHERE index_timestamp >= NOW() - INTERVAL '{days} days'
        ORDER BY index_timestamp
    """
    return pd.read_sql(query, conn)


def fetch_funding_rates(conn, days: int = 90) -> pd.DataFrame:
    """Fetch funding rates per symbol (last N days)."""
    query = f"""
        SELECT
            coin_symbol as symbol,
            funding_time as timestamp,
            funding_rate
        FROM funding_rates
        WHERE funding_time >= NOW() - INTERVAL '{days} days'
        ORDER BY coin_symbol, funding_time
    """
    df = pd.read_sql(query, conn)
    # Normalize symbol to match historical_klines format (add USDT if needed)
    df['symbol'] = df['symbol'].apply(lambda x: x if x.endswith('USDT') else f"{x}USDT")
    return df


def fetch_news_sentiment(conn, days: int = 90) -> pd.DataFrame:
    """Fetch news sentiment per symbol (last N days)."""
    query = f"""
        SELECT
            coin_symbol as symbol,
            recorded_at as timestamp,
            net_sentiment as sentiment_score
        FROM news_sentiment
        WHERE recorded_at >= NOW() - INTERVAL '{days} days'
        ORDER BY coin_symbol, recorded_at
    """
    df = pd.read_sql(query, conn)
    df['symbol'] = df['symbol'].apply(lambda x: x if x.endswith('USDT') else f"{x}USDT")
    return df


def fetch_btc_prices(conn, days: int = 90) -> pd.DataFrame:
    """Fetch BTC prices for calculating BTC price change (last N days)."""
    query = f"""
        SELECT
            open_time as timestamp,
            close_price as btc_close
        FROM historical_klines
        WHERE symbol = 'BTCUSDT'
          AND open_time >= NOW() - INTERVAL '{days} days'
        ORDER BY open_time
    """
    return pd.read_sql(query, conn)


def build_features_dataframe(conn, days: int = 90, significant_moves: bool = False, threshold: float = 0.5) -> pd.DataFrame:
    """Build the complete features dataframe (last N days of data).

    Args:
        conn: Database connection
        days: Number of days of data to fetch
        significant_moves: If True, use significant move labeling (LONG/SHORT/exclude)
        threshold: Percentage threshold for significant moves (default 0.5%)
    """
    print(f"Fetching last {days} days of data...")

    datasets = {}
    with tqdm(total=5, desc="Fetching data", unit="table") as pbar:
        pbar.set_description("Fetching historical klines")
        datasets['df'] = fetch_historical_klines(conn, days)
        pbar.update(1)

        pbar.set_description("Fetching fear & greed index")
        datasets['fear_greed'] = fetch_fear_greed(conn, days)
        pbar.update(1)

        pbar.set_description("Fetching funding rates")
        datasets['funding'] = fetch_funding_rates(conn, days)
        pbar.update(1)

        pbar.set_description("Fetching news sentiment")
        datasets['sentiment'] = fetch_news_sentiment(conn, days)
        pbar.update(1)

        pbar.set_description("Fetching BTC prices")
        datasets['btc'] = fetch_btc_prices(conn, days)
        pbar.update(1)

    df = datasets['df']
    fear_greed = datasets['fear_greed']
    funding = datasets['funding']
    sentiment = datasets['sentiment']
    btc = datasets['btc']

    print(f"\nRows fetched: klines={len(df)}, fear_greed={len(fear_greed)}, "
          f"funding={len(funding)}, sentiment={len(sentiment)}, btc={len(btc)}")

    # Convert timestamps to timezone-naive for merging
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    fear_greed['timestamp'] = pd.to_datetime(fear_greed['timestamp']).dt.tz_localize(None)
    funding['timestamp'] = pd.to_datetime(funding['timestamp']).dt.tz_localize(None)
    sentiment['timestamp'] = pd.to_datetime(sentiment['timestamp']).dt.tz_localize(None)
    btc['timestamp'] = pd.to_datetime(btc['timestamp']).dt.tz_localize(None)

    # Fear & Greed is daily data - merge on date (not 4-hour timestamp)
    fear_greed['date'] = fear_greed['timestamp'].dt.date
    fear_greed = fear_greed.drop(columns=['timestamp']).drop_duplicates(subset=['date'], keep='last')
    df['date'] = df['timestamp'].dt.date

    # Round funding and sentiment timestamps to 4 hours
    funding['timestamp'] = funding['timestamp'].dt.floor('4h')
    funding = funding.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')

    sentiment['timestamp'] = sentiment['timestamp'].dt.floor('4h')
    sentiment = sentiment.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')

    # Calculate BTC price change (percentage change from previous period)
    btc = btc.sort_values('timestamp')
    btc['price_change_btc'] = btc['btc_close'].pct_change() * 100
    btc = btc[['timestamp', 'price_change_btc']]

    # Merge datasets with progress bar
    def merge_fear_greed(d):
        merged = d.merge(fear_greed, on='date', how='left')
        return merged.drop(columns=['date'])

    merge_steps = [
        ("Merging fear & greed", merge_fear_greed),
        ("Merging funding rates", lambda d: d.merge(funding, on=['symbol', 'timestamp'], how='left')),
        ("Merging sentiment", lambda d: d.merge(sentiment, on=['symbol', 'timestamp'], how='left')),
        ("Merging BTC price change", lambda d: d.merge(btc, on='timestamp', how='left')),
    ]

    with tqdm(total=len(merge_steps) + 1, desc="Processing", unit="step") as pbar:
        for name, merge_func in merge_steps:
            pbar.set_description(name)
            df = merge_func(df)
            pbar.update(1)

        # Create target variable
        pbar.set_description("Creating target variable")
        df = df.sort_values(['symbol', 'timestamp'])

        if significant_moves:
            # Significant moves labeling: predict LONG (1) or SHORT (0) opportunities
            # Get next candle's OHLC
            df['next_open'] = df.groupby('symbol')['open'].shift(-1)
            df['next_high'] = df.groupby('symbol')['high'].shift(-1)
            df['next_low'] = df.groupby('symbol')['low'].shift(-1)

            # Calculate percentage moves from next candle's open
            df['high_pct'] = (df['next_high'] - df['next_open']) / df['next_open'] * 100
            df['low_pct'] = (df['next_open'] - df['next_low']) / df['next_open'] * 100

            # Label based on significant moves
            # 1 = LONG opportunity (high hits threshold, low doesn't)
            # 0 = SHORT opportunity (low hits threshold, high doesn't)
            # NaN = exclude (ambiguous or no move)
            def label_move(row):
                high_hit = row['high_pct'] >= threshold
                low_hit = row['low_pct'] >= threshold
                if high_hit and not low_hit:
                    return 1.0  # LONG
                elif low_hit and not high_hit:
                    return 0.0  # SHORT
                else:
                    return float('nan')  # Exclude (both hit or neither)

            df['target'] = df.apply(label_move, axis=1)

            # Drop helper columns
            df = df.drop(columns=['next_open', 'next_high', 'next_low', 'high_pct', 'low_pct'])

            # Count before dropping
            total_rows = len(df)
            excluded = df['target'].isna().sum()
            long_count = (df['target'] == 1.0).sum()
            short_count = (df['target'] == 0.0).sum()

            # Drop rows with NaN target (ambiguous/no move)
            df = df.dropna(subset=['target'])

            print(f"\n  Significant moves labeling (threshold: {threshold}%):")
            print(f"    LONG opportunities: {long_count} ({100*long_count/total_rows:.1f}%)")
            print(f"    SHORT opportunities: {short_count} ({100*short_count/total_rows:.1f}%)")
            print(f"    Excluded (ambiguous/no move): {excluded} ({100*excluded/total_rows:.1f}%)")
        else:
            # Original labeling: 1 if next period's close > current close, 0 otherwise
            df['next_close'] = df.groupby('symbol')['close'].shift(-1)
            df['target'] = (df['next_close'] > df['close']).astype(float)
            df = df.drop(columns=['next_close'])

        pbar.update(1)

    # Reorder columns
    columns = [
        'timestamp', 'symbol', 'open', 'high', 'low', 'close',
        'rsi_14', 'rsi_30', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_width',
        'momentum_10', 'stoch_k', 'stoch_d',
        'volume', 'volume_sma',
        'funding_rate', 'fear_greed', 'sentiment_score',
        'price_change_btc', 'target'
    ]
    df = df[columns]

    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    # Data quality summary
    indicator_cols = ['rsi_14', 'rsi_30', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                      'bb_width', 'momentum_10', 'stoch_k', 'stoch_d', 'volume_sma']
    rows_with_indicators = df[indicator_cols].notna().all(axis=1).sum()
    print(f"Rows with all indicators: {rows_with_indicators} ({100*rows_with_indicators/len(df):.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build features CSV from database for Boruta analysis."
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=90,
        help="Number of days of data to fetch (default: 90)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename (default: boruta-features-YYYY-MM-DD.csv)"
    )
    parser.add_argument(
        "--significant-moves", "-s",
        action="store_true",
        help="Use significant moves labeling instead of directional (excludes ambiguous candles)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Percentage threshold for significant moves (default: 0.5)"
    )
    args = parser.parse_args()

    output_file = args.output or f"boruta-features-{datetime.now().strftime('%Y-%m-%d')}.csv"

    print("Connecting to database...")
    conn = get_connection()

    try:
        df = build_features_dataframe(
            conn,
            days=args.days,
            significant_moves=args.significant_moves,
            threshold=args.threshold
        )

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")

        # Show sample (recent data with indicators)
        print("\nSample data (recent rows with indicators):")
        sample = df[df['rsi_14'].notna()].tail(3)
        print(sample.to_string())

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
