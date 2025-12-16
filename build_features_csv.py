#!/usr/bin/env python3
"""
Build features CSV from Neon database for Boruta analysis.

Pulls data from historical_klines, fear_greed_index, funding_rates,
and news_sentiment tables, then creates a target variable based on
whether the price goes up in the next period.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm


def get_connection():
    """Get database connection from .env.local."""
    env_path = Path(__file__).parent / '.env.local'
    load_dotenv(env_path)

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not found in .env.local")

    return psycopg2.connect(database_url)


def fetch_historical_klines(conn) -> pd.DataFrame:
    """Fetch historical klines with technical indicators."""
    query = """
        SELECT
            symbol,
            open_time as timestamp,
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
        ORDER BY symbol, open_time
    """
    return pd.read_sql(query, conn)


def fetch_fear_greed(conn) -> pd.DataFrame:
    """Fetch fear & greed index data."""
    query = """
        SELECT
            index_timestamp as timestamp,
            value as fear_greed
        FROM fear_greed_index
        ORDER BY index_timestamp
    """
    return pd.read_sql(query, conn)


def fetch_funding_rates(conn) -> pd.DataFrame:
    """Fetch funding rates per symbol."""
    query = """
        SELECT
            coin_symbol as symbol,
            funding_time as timestamp,
            funding_rate
        FROM funding_rates
        ORDER BY coin_symbol, funding_time
    """
    df = pd.read_sql(query, conn)
    # Normalize symbol to match historical_klines format (add USDT if needed)
    df['symbol'] = df['symbol'].apply(lambda x: x if x.endswith('USDT') else f"{x}USDT")
    return df


def fetch_news_sentiment(conn) -> pd.DataFrame:
    """Fetch news sentiment per symbol."""
    query = """
        SELECT
            coin_symbol as symbol,
            recorded_at as timestamp,
            net_sentiment as sentiment_score
        FROM news_sentiment
        ORDER BY coin_symbol, recorded_at
    """
    df = pd.read_sql(query, conn)
    df['symbol'] = df['symbol'].apply(lambda x: x if x.endswith('USDT') else f"{x}USDT")
    return df


def fetch_btc_prices(conn) -> pd.DataFrame:
    """Fetch BTC prices for calculating BTC price change."""
    query = """
        SELECT
            open_time as timestamp,
            close_price as btc_close
        FROM historical_klines
        WHERE symbol = 'BTCUSDT'
        ORDER BY open_time
    """
    return pd.read_sql(query, conn)


def build_features_dataframe(conn) -> pd.DataFrame:
    """Build the complete features dataframe."""
    steps = [
        ("Fetching historical klines", fetch_historical_klines),
        ("Fetching fear & greed index", fetch_fear_greed),
        ("Fetching funding rates", fetch_funding_rates),
        ("Fetching news sentiment", fetch_news_sentiment),
        ("Fetching BTC prices", fetch_btc_prices),
    ]

    datasets = {}
    with tqdm(total=len(steps), desc="Fetching data", unit="table") as pbar:
        for name, fetch_func in steps:
            pbar.set_description(name)
            if fetch_func == fetch_historical_klines:
                datasets['df'] = fetch_func(conn)
            elif fetch_func == fetch_fear_greed:
                datasets['fear_greed'] = fetch_func(conn)
            elif fetch_func == fetch_funding_rates:
                datasets['funding'] = fetch_func(conn)
            elif fetch_func == fetch_news_sentiment:
                datasets['sentiment'] = fetch_func(conn)
            elif fetch_func == fetch_btc_prices:
                datasets['btc'] = fetch_func(conn)
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

    # Round timestamps to nearest 4 hours for fear_greed merge
    fear_greed['timestamp'] = fear_greed['timestamp'].dt.floor('4H')
    fear_greed = fear_greed.drop_duplicates(subset=['timestamp'], keep='last')

    # Round funding and sentiment timestamps
    funding['timestamp'] = funding['timestamp'].dt.floor('4H')
    funding = funding.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')

    sentiment['timestamp'] = sentiment['timestamp'].dt.floor('4H')
    sentiment = sentiment.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')

    # Calculate BTC price change (percentage change from previous period)
    btc = btc.sort_values('timestamp')
    btc['price_change_btc'] = btc['btc_close'].pct_change() * 100
    btc = btc[['timestamp', 'price_change_btc']]

    # Merge datasets with progress bar
    merge_steps = [
        ("Merging fear & greed", lambda d: d.merge(fear_greed, on='timestamp', how='left')),
        ("Merging funding rates", lambda d: d.merge(funding, on=['symbol', 'timestamp'], how='left')),
        ("Merging sentiment", lambda d: d.merge(sentiment, on=['symbol', 'timestamp'], how='left')),
        ("Merging BTC price change", lambda d: d.merge(btc, on='timestamp', how='left')),
    ]

    with tqdm(total=len(merge_steps) + 1, desc="Processing", unit="step") as pbar:
        for name, merge_func in merge_steps:
            pbar.set_description(name)
            df = merge_func(df)
            pbar.update(1)

        # Create target: 1 if next period's close > current close, 0 otherwise
        pbar.set_description("Creating target variable")
        df = df.sort_values(['symbol', 'timestamp'])
        df['next_close'] = df.groupby('symbol')['close'].shift(-1)
        df['target'] = (df['next_close'] > df['close']).astype(float)
        pbar.update(1)

    # Drop the helper column and last row per symbol (no target)
    df = df.drop(columns=['next_close'])

    # Reorder columns
    columns = [
        'timestamp', 'symbol', 'close',
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

    return df


def main():
    output_file = f"boruta-features-{datetime.now().strftime('%Y-%m-%d')}.csv"

    print("Connecting to database...")
    conn = get_connection()

    try:
        df = build_features_dataframe(conn)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")

        # Show sample
        print("\nSample data:")
        print(df.head(3).to_string())

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
