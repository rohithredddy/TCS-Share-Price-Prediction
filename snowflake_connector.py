# snowflake_connector.py
import os
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

def get_sf_conn(creds=None):
    """
    Return a Snowflake connection using provided creds dict or environment variables.
    creds expected keys: user, password, account, warehouse, database, schema, role (optional)
    """
    if creds is None:
        creds = {
            'user': os.environ.get('SNOWFLAKE_USER'),
            'password': os.environ.get('SNOWFLAKE_PASSWORD'),
            'account': os.environ.get('SNOWFLAKE_ACCOUNT'),
            'warehouse': os.environ.get('SNOWFLAKE_WAREHOUSE'),
            'database': os.environ.get('SNOWFLAKE_DATABASE'),
            'schema': os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC'),
            'role': os.environ.get('SNOWFLAKE_ROLE')
        }
    conn = snowflake.connector.connect(
        user=creds.get('user'),
        password=creds.get('password'),
        account=creds.get('account'),
        warehouse=creds.get('warehouse'),
        database=creds.get('database'),
        schema=creds.get('schema'),
        role=creds.get('role')
    )
    return conn

def write_predictions_to_snowflake(df, table_name='TCS_PREDICTIONS', creds=None):
    """
    Write predictions dataframe to Snowflake.
    df must include columns: 'Date' and 'Predicted_Close'
    Returns True on success, False on failure.
    """
    # Validate
    if 'Date' not in df.columns or 'Predicted_Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Date' and 'Predicted_Close' columns")

    # Prepare table-ready dataframe
    df_to_write = df.copy()
    df_to_write['DATE'] = pd.to_datetime(df_to_write['Date']).dt.date
    df_to_write = df_to_write[['DATE', 'Predicted_Close']].rename(columns={'Predicted_Close':'PREDICTED_CLOSE'})

    conn = get_sf_conn(creds)
    try:
        cs = conn.cursor()
        # Create table if not exists
        cs.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                DATE DATE,
                PREDICTED_CLOSE FLOAT
            )
        """)
        cs.close()

        # Use write_pandas for efficient upload
        success, nchunks, nrows, _ = write_pandas(conn, df_to_write, table_name)
        conn.close()
        return bool(success)
    except Exception as e:
        # Ensure connection closed and return False
        try:
            conn.close()
        except:
            pass
        print("Snowflake write failed:", e)
        return False
