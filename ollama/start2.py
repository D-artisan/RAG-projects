import ollama


res = ollama.chat(
    model="gemma3:27b", 
    messages=[
        {"role": "user", "content": "What is household policy?"}
    ],
    stream=True
)

for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)


# ==================================================================================
# ==== The Ollama Python library's API is designed around the Ollama REST API ====
# ==================================================================================
