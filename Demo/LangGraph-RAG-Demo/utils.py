from langchain_google_genai import ChatGoogleGenerativeAI
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