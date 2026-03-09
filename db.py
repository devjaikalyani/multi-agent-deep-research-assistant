"""
Database utility
- Creates and seeds a demo SQLite database with sample market data
- Data Agent uses this for live SQL queries
- Add your own CSV files to /data to extend it
"""

import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/research.db")


def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _seed_if_empty(conn)
    return conn


def _seed_if_empty(conn: sqlite3.Connection):
    """Seed the database with demo market data if tables don't exist yet."""
    cursor = conn.cursor()

    # Check if already seeded
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    if tables:
        return

    print("🌱 Seeding demo database...")

    # EV Market Data
    ev_data = pd.DataFrame({
        "country": ["India", "China", "USA", "Germany", "UK", "France", "Japan", "South Korea"],
        "ev_sales_2023": [80000, 6200000, 1200000, 520000, 310000, 280000, 89000, 162000],
        "ev_sales_2024": [130000, 7800000, 1450000, 580000, 370000, 310000, 105000, 190000],
        "market_share_pct": [2.1, 35.7, 9.5, 22.3, 18.6, 21.4, 3.8, 11.2],
        "yoy_growth_pct": [62.5, 25.8, 20.8, 11.5, 19.4, 10.7, 18.0, 17.3]
    })
    ev_data.to_sql("ev_market", conn, if_exists="replace", index=False)

    # India EV Manufacturers
    india_ev = pd.DataFrame({
        "manufacturer": ["Tata Motors", "MG Motor", "Hyundai", "Kia", "Mahindra", "BYD", "Ola Electric"],
        "market_share_pct": [38.5, 12.3, 11.7, 8.4, 9.1, 5.2, 14.8],
        "models_available": [4, 2, 2, 1, 3, 2, 3],
        "avg_price_inr_lakhs": [14.5, 24.0, 23.8, 29.5, 21.9, 26.9, 12.5],
        "range_km": [350, 461, 484, 528, 375, 420, 181]
    })
    india_ev.to_sql("india_ev_manufacturers", conn, if_exists="replace", index=False)

    # Global AI Market
    ai_market = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027],
        "market_size_bn_usd": [62.4, 87.0, 119.8, 166.0, 229.0, 305.9, 407.0, 543.0],
        "yoy_growth_pct": [None, 39.4, 37.7, 38.6, 37.9, 33.6, 33.1, 33.4],
        "top_segment": ["ML", "NLP", "CV", "Generative AI", "Generative AI", "Agentic AI", "Agentic AI", "Agentic AI"]
    })
    ai_market.to_sql("ai_market", conn, if_exists="replace", index=False)

    # Cloud Provider Comparison
    cloud_data = pd.DataFrame({
        "provider": ["AWS", "Azure", "GCP", "Alibaba Cloud", "Oracle Cloud"],
        "market_share_pct": [31.0, 25.0, 11.0, 4.0, 2.0],
        "revenue_2024_bn_usd": [107.6, 98.3, 43.2, 15.7, 8.4],
        "yoy_growth_pct": [17.2, 29.0, 35.2, 7.8, 24.1],
        "ml_services_count": [75, 60, 55, 32, 28]
    })
    cloud_data.to_sql("cloud_providers", conn, if_exists="replace", index=False)

    # Healthcare AI
    health_ai = pd.DataFrame({
        "application": ["Medical Imaging", "Drug Discovery", "Clinical Documentation", "Patient Monitoring", "Administrative AI"],
        "market_size_2024_bn": [4.1, 3.8, 2.9, 2.3, 1.8],
        "market_size_2030_bn": [18.3, 16.1, 9.4, 8.7, 6.2],
        "cagr_pct": [28.4, 27.0, 21.6, 24.8, 22.8],
        "adoption_rate_pct": [38, 24, 61, 29, 52]
    })
    health_ai.to_sql("healthcare_ai", conn, if_exists="replace", index=False)

    conn.commit()
    print(f"✅ Database seeded with {len(tables)+5} tables at {DB_PATH}")


def get_db_schema() -> str:
    """Return a description of all tables and columns for SQL generation."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_names = [f"{c[1]} ({c[2]})" for c in cols]

        # Sample row for context
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        sample = cursor.fetchone()

        schema_parts.append(
            f"Table: {table}\n"
            f"Columns: {', '.join(col_names)}\n"
            f"Sample row: {sample}"
        )

    conn.close()
    return "\n\n".join(schema_parts)


def load_csv_to_db(csv_path: str, table_name: str = None):
    """
    Load any CSV file into the SQLite database.
    Usage: load_csv_to_db("data/my_data.csv", "my_table")
    """
    if not table_name:
        table_name = os.path.splitext(os.path.basename(csv_path))[0].lower().replace(" ", "_")

    df = pd.read_csv(csv_path)
    conn = get_db_connection()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Loaded {len(df)} rows into table '{table_name}'")
    return table_name
