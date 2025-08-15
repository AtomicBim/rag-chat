QDRANT_HOST = "192.168.42.188"
QDRANT_PORT = 6333
COLLECTION_NAME = "internal_regulations_v2"
SEARCH_LIMIT = 30  # Лимит документов для проверки ответа

EMBEDDING_SERVICE_ENDPOINT = "http://192.168.45.55:8001/create_embedding" 
OPENAI_API_ENDPOINT = "http://192.168.45.55:8000/generate_answer" 

DOCS_ENDPOINT = "http://192.168.45.55:8001/documents/"