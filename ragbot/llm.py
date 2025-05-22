from transformers import pipeline
from django.conf import settings
import torch
import os


def truncate_text(text, max_tokens=600):
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + "\n\n[Context Truncated]"

# Check environment variable to optionally force CPU
force_cpu = os.getenv("FORCE_CPU", "0") == "1"

# Select device based on MPS availability and environment variable
if not force_cpu and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# HuggingFace pipeline expects device as int:
#   -1 for CPU
#    0,1,... for GPU devices
if device.type == "cuda":
    device_id = 0
elif device.type == "mps":
    device_id = "mps"
else:
    device_id = -1

# Initialize text generation pipeline with selected device
generator = pipeline(
    "text-generation",
    model=settings.LLM_MODEL,
    token=settings.HUGGINGFACE_TOKEN,
    pad_token_id=50256,
    device=device_id
)

def generate_response(context_chunks, question):
    if isinstance(context_chunks[0], tuple):
        context_with_source = [(file, truncate_text(text)) for file, text in context_chunks]
        context = "\n\n".join([f"From file: {file}\n{text}" for file, text in context_with_source])
        sources = list({file for file, _ in context_chunks})
    else:
        context = truncate_text("\n\n".join(context_chunks))
        sources = []

    prompt = f"""
        You are a highly reliable assistant trained to answer any question using internal company documents such as training agendas, engineering workflows, best practices, and business procedures.

        Use only the information from the provided context below to answer the user's question. If the answer cannot be derived directly from the context, respond with:

        "The available documents do not contain enough information to confidently answer this question."

        Always respond clearly and professionally, and include the file names you used as references in a "Sources" section at the end.

        ---
        Context:
        {context}

        ---
        Question:
        {question}

        ---
        Answer:
    """

    output = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]['generated_text']

    if "Answer:" in output:
        answer = output.split("Answer:")[-1].strip()
    else:
        answer = output.strip()

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
        "question": question,
        "prompt": prompt.strip(),
        "raw_output": output.strip()
    }
