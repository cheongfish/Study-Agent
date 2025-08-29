import os
from psycopg2.extras import execute_values
import pandas as pd
from dotenv import load_dotenv
from Utils import postgres_conn
from Utils import JsonEmbedder


class PostgresEmbeddingLoader:
    """
    A class to handle embedding loading processes into a PostgreSQL database with pgvector.
    """
    def __init__(self):
        """
        Initializes the database connection and enables the pgvector extension.
        """
        self.conn = postgres_conn()
        self.cursor = self.conn.cursor()
        self._activate_pgvector()

    def _activate_pgvector(self):
        """Activates the pgvector extension in the database."""
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()
        print("pgvector extension is enabled.")

    def _map_dtype_to_sql(
        self,
        dtype,
        col_name,
        vector_col,
        vector_dim,
        ):
        """Maps pandas dtype to a corresponding PostgreSQL data type."""
        if col_name == vector_col:
            return f"VECTOR({vector_dim})"
        if pd.api.types.is_integer_dtype(dtype):
            return "INT"
        elif pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        elif pd.api.types.is_string_dtype(dtype) or dtype == 'object':
            return "TEXT"
        else:
            return "TEXT"  # Default for other types

    def generate_create_table_sql(
        self,
        table_name,
        df,
        vector_col='embedding',
        vector_dim=1024,
        unique_col=None,
        ):
        """
        Dynamically generates a CREATE TABLE SQL statement from a pandas DataFrame.
        """
        columns = ["id SERIAL PRIMARY KEY"]
        for col, dtype in df.dtypes.items():
            # Handle SQL reserved words like 'order'
            col_name = 'order_val' if col.lower() == 'order' else col
            
            sql_type = self._map_dtype_to_sql(dtype, col, vector_col, vector_dim)
            
            if col_name == unique_col:
                columns.append(f"{col_name} {sql_type} UNIQUE")
            else:
                columns.append(f"{col_name} {sql_type}")

        return f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});"

    def create_table(
        self,
        table_name,
        df,
        vector_col='embedding',
        vector_dim=1024,
        unique_col=None,
        ):
        """Creates a table in the database based on the DataFrame structure."""
        create_sql = self.generate_create_table_sql(table_name, df, vector_col, vector_dim, unique_col)
        print(f"Executing: {create_sql}")
        self.cursor.execute(create_sql)
        self.conn.commit()
        print(f"Table '{table_name}' created successfully.")

    def insert_data(
        self,
        table_name,
        df,
        unique_col,
        ):
        """Inserts data from a DataFrame into the specified table."""
        # Replace NaN with None for NULL insertion in DB
        df = df.where(pd.notnull(df), None)

        # Prepare column names, handling reserved words like 'order'
        cols = ['order_val' if col.lower() == 'order' else col for col in df.columns]

        # Constructing the insert query
        insert_query = f"""
        INSERT INTO {table_name} ({', '.join(cols)})
        VALUES %s
        ON CONFLICT ({unique_col}) DO NOTHING;
        """

        # Convert DataFrame to a list of tuples for insertion
        data_tuples = [tuple(row) for row in df.itertuples(index=False)]

        print(f"Inserting {len(data_tuples)} rows into '{table_name}'...")
        execute_values(self.cursor, insert_query, data_tuples)
        self.conn.commit()
        print("Data insertion complete.")

    def close_connection(self):
        """Closes the database cursor and connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Database connection closed.")

if __name__ == '__main__':
    pwd = os.getcwd()
    env_path = pwd + "/.env"
    load_dotenv(env_path) 
    # --- Example Usage ---
    # Before running, ensure you have set your Google API key as an environment variable.
    # In your terminal, run:
    # export GOOGLE_API_KEY="YOUR_API_KEY"

    try:
        # 1. Initialize the embedder
        model_name = "BAAI/bge-m3"
        
        # 2. Define input/output paths and the column to embed
        #    Please CHANGE these paths to your actual file paths.
        input_json_path = pwd + '/all_basecode.json'
        output_json_path_tpl = pwd + "/all_basecode_embeddings_{model_name}.json"
        column_to_embed = 'content'  # The name of the column you want to embed
        
        embedder = JsonEmbedder(model_name,output_json_path_tpl)
        output_json_path = embedder.concated_outpath

        print("\n--- Starting Embedding Process ---")
        print(f"Input file: {input_json_path}")
        print(f"Output file: {output_json_path}")
        print(f"Column to embed: {column_to_embed}")
        print("------------------------------------")

        # 3. Run the process
        # Note: This example will not run if the input file does not exist.
        if os.path.exists(output_json_path):
            print("Already embedded file exists skip to load on pgvector")
            
        else:
            try:
                embedder.process_file(
                    input_path=input_json_path,
                    column_to_embed=column_to_embed
                )
            except FileNotFoundError:
                print(f"Example skipped: Input file '{input_json_path}' not found.")
                print("Please update the 'input_json_path' variable in the script to point to your file.")
                raise

    except ValueError as e:
        print(f"Error: {e}")
    
    # Example usage of the PostgresEmbeddingLoader class
    loader = PostgresEmbeddingLoader()

    try:
        print("Start insert embedded file to pgvector")
        # 1. Load data from a JSON file into a pandas DataFrame
        df = pd.read_json(output_json_path)
        print(f"Loaded {len(df)} records from '{output_json_path}'")

        # 2. Dynamically create a table from the DataFrame
        table_name = 'curriculum'
        # Determine vector dimension from the first embedding vector
        vector_dim = len(df['embedding'][0]) if not df.empty and 'embedding' in df.columns and len(df['embedding']) > 0 else 1024
        
        loader.create_table(
            table_name=table_name,
            df=df,
            vector_col='embedding',
            vector_dim=vector_dim,
            unique_col='basecode'
        )

        # 3. Insert the data into the newly created table
        loader.insert_data(table_name=table_name, df=df, unique_col='basecode')

    except FileNotFoundError:
        print(f"Error: The file '{output_json_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 4. Close the database connection
        loader.close_connection()
