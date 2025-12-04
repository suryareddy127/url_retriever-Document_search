import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model

load_dotenv()

class config:
    """configuration class for RAG"""
    # api key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # model configuration
    LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"

    # document processing 
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # default urls
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://www.chitika.com/open-source-models-rag/" 
    ]

    @classmethod
    def get_llm(cls):
        """initialize and return the llm model"""
        os.environ["DEEPSEEK_API_KEY"] =cls.DEEPSEEK_API_KEY
        return init_chat_model(cls.LLM_MODEL)
