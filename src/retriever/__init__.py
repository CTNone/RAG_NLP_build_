import warnings

# Tắt cảnh báo LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.retriever.document_retriever import (
    get_data_directories,
    load_json_documents,
    retrieve_documents,
)

__all__ = ["retrieve_documents", "load_json_documents", "get_data_directories"]
