import ollama
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag") # model_name="all-MiniLM-L6-v2"
collection.add(
    documents=[
        "Alex loves dynamically typed languages",
        "Philip likes statically typed languages"
    ],
    ids=["id1", "id2"]
)

while(True):
    question = input(">>> ") 
    if question == "exit": exit()

    # Retrieval
    results = collection.query(
        query_texts=[question],
        n_results=1
    )
    context = results["documents"][0][0]

    # Augmentation
    template = f"Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"

    # Generation
    stream = ollama.chat(
        model='llama3',
        messages=[{'role': 'system', 'content': template}],
        stream=True,
    )
    for chunk in stream:
      print(chunk['message']['content'], end='', flush=True)
    print()
