from langchain_community.cache import InMemoryCache  # Updated import
from langchain_openai import ChatOpenAI
from src.config import DEFAULT_MODEL
import langchain_core
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
import os

# Set BaseCache globally
langchain_core.caches.BaseCache = InMemoryCache

# Call model_rebuild() at module level
ChatOpenAI.model_rebuild()

# Create a singleton ChatOpenAI instance
def get_chat_model() -> ChatOpenAI:
    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return model

chat_model = get_chat_model()