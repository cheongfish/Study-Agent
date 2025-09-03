from langchain_google_genai import ChatGoogleGenerativeAI
import psycopg2
from Settings import DB_PARAMS
from Settings import env_path
import os
from dotenv import load_dotenv
import pandas as pd
from google.genai import Client
from tqdm import tqdm
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv(env_path)
def init_llm():
    # define llm
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    return llm

def postgres_conn():
    conn = psycopg2.connect(**DB_PARAMS)
    return conn



class JsonEmbedder:
    """
    A class to read a JSON file, embed a specific column using a specified Gemini embedding model,
    and save the result to a new JSON file.
    """
    def __init__(
        self,
        model_name,
        output_path_tpl,
        api_key=None,
        ):
        """
        Initializes the embedder and configures the Gemini API.

        Args:
            model_name (str): The name of the embedding model to use.
            output_path_tpl(str): The path of output path teplate
            api_key (str, optional): Your Google API key. If not provided,
                                     it will be read from the GOOGLE_API_KEY
                                     environment variable. Defaults to None.
        """

        self.api_key = api_key
        self.model_name = model_name
        self._init_model()
        self.concated_outpath = output_path_tpl.format(
            model_name=self.result_postfix,
        )
        print(f"Using embedding model: {self.model_name}")
        
    def _init_model(self):
        
        if 'gemini' in self.model_name:
            if self.api_key:
                self.model = Client(api_key=self.api_key)
            elif "GOOGLE_API_KEY" in os.environ:
                self.model = Client(api_key=os.environ["GOOGLE_API_KEY"])
            else:
                raise ValueError("Google API Key not provided or set in GOOGLE_API_KEY environment variable.")
            self.result_postfix = self.model_name
        else:
            self.model = SentenceTransformer(self.model_name)
            self.result_postfix = self.model_name.split("/")[-1]
            
    def _embed_texts(
        self,
        texts,
        embedding_dim=1024
        ):
        """
        Embeds a list of texts using the served model

        Args:
            texts (list[str]): A list of strings to embed.

        Returns:
            list: A list of embedding vectors.
        """
        try:
            if "gemini" in self.model_name:
                _embeddings = self.model.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=embedding_dim)
                ).embeddings
                embeddings = [embedding.values for embedding in _embeddings]
                
            else:
                embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"An error occurred during embedding: {e}")
            return [None] * len(texts)
        
    def process_file(
        self,
        input_path,
        column_to_embed,
        batch_size=50,
        ):
        """
        Reads a JSON file, embeds a specified column, and saves the result.

        Args:
            input_path (str): Path to the input JSON file.
            column_to_embed (str): The name of the column in the JSON file to embed.
            batch_size (int, optional): The number of texts to embed in each API call.
                                       Defaults to 100.
        """
        print(f"Reading data from {input_path}...")
        try:
            df = pd.read_json(input_path)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_path}")
            return
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return

        
        texts_to_embed = df[column_to_embed].astype(str).tolist()
        all_embeddings = []

        print(f"Embedding column '{column_to_embed}' in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts_to_embed), batch_size)):
            batch = texts_to_embed[i:i + batch_size]
            embeddings = self._embed_texts(batch)
                
            all_embeddings.extend(embeddings)

        df['embedding'] = all_embeddings

        # Ensure the output directory exists
        output_dir = os.path.dirname(self.concated_outpath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Saving data with embeddings to {self.concated_outpath}...")
        df.to_json(
            self.concated_outpath,
            orient='records',
            indent=4,
            force_ascii=False,
        )
        print("Processing complete.")