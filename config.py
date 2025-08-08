QDRANT_HOST = "qdrant"  # Qdrant работает на той же машине, можно использовать localhost
QDRANT_PORT = 6333
COLLECTION_NAME = "internal_regulations_v2"
SEARCH_LIMIT = 5 # Лимит для более  релевантных ответов

# IP вашего ПК с GPU, где запущен embedding_service.py
EMBEDDING_SERVICE_ENDPOINT = "http://192.168.45.55:8001/create_embedding" 
# IP вашей ВМ, где запущен ask_question.py
OPENAI_API_ENDPOINT = "http://192.168.45.79:8000/generate_answer" 
