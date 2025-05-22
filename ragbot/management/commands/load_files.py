from django.core.management.base import BaseCommand
from ragbot.file_loader import load_file_text, chunk_text
from ragbot.embedder import embed_texts
from ragbot.retriever import build_index
import os

class Command(BaseCommand):
    help = 'Reads and embeds all supported files from /files folder'

    def handle(self, *args, **kwargs):
        folder = "files"
        all_chunks = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            try:
                text = load_file_text(path)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

        embeddings = embed_texts(all_chunks)
        build_index(embeddings, all_chunks)
        print("Files embedded and index saved.")