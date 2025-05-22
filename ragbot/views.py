from django.http import JsonResponse
from rest_framework.decorators import api_view
from ragbot.embedder import embed_texts
from ragbot.retriever import load_index, query_index
from ragbot.llm import generate_response

load_index()

@api_view(["GET"])
def ask(request):
    question = request.GET.get("q")
    if not question:
        return JsonResponse({"error": "Missing query"}, status=400)

    query_embedding = embed_texts([question])[0]
    context_chunks = query_index(query_embedding)
    context = "\n".join(context_chunks)

    answer = generate_response(context, question)
    
    return JsonResponse({"question": question, "answer": answer})