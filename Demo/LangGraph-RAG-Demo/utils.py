from langchain_google_genai import ChatGoogleGenerativeAI
import psycopg2
from settings import DB_PARAMS
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