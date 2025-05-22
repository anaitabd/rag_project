from sentence_transformers import SentenceTransformer
from django.conf import settings

embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def embed_texts(texts):
    return embed_model.encode(texts, convert_to_numpy=True)