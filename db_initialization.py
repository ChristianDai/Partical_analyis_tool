import streamlit as st
from sqlalchemy import text

# Create the SQL connection to symbolic_regression_db as specified in your secrets file.
conn = st.connection('symbolic_regression_db', type='sql')

def initialize_database():
    sql_create_runs_table = """ CREATE TABLE IF NOT EXISTS runs (
                                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    model_name TEXT NOT NULL,
                                    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    r2 REAL,
                                    rmse REAL,
                                    mae REAL,
                                    mape REAL
                                ); """

    sql_create_dataset_details_table = """ CREATE TABLE IF NOT EXISTS dataset_details (
                                                run_id INTEGER,
                                                num_samples INTEGER,
                                                num_features INTEGER,
                                                num_missing_values INTEGER,
                                                biggest_dependent_value REAL,
                                                smallest_dependent_value REAL,
                                                smallest_absolute_dependent_value REAL,
                                                biggest_independent_value REAL,
                                                smallest_independent_value REAL,
                                                smallest_absolute_independent_value REAL,
                                                dependent_mean REAL,
                                                dependent_std_dev REAL,
                                                dependent_quartile_1 REAL,
                                                dependent_median REAL,
                                                dependent_quartile_3 REAL,
                                                dependent_geometric_mean REAL,
                                                dependent_geometric_std_dev REAL,
                                                dependent_outliers_proportion REAL,
                                                dependent_skewness REAL,
                                                dependent_kurtosis REAL,
                                                FOREIGN KEY (run_id) REFERENCES runs(run_id)
                                            );"""

    # Create tables using the connection session
    with conn.session as s:
        try:
            s.execute(text(sql_create_runs_table))
            s.execute(text(sql_create_dataset_details_table))
            s.commit()
        except Exception as e:
            st.error(f"Error: {e}")
            s.rollback()