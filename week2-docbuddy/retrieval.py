from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import torch
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if torch.cuda.is_available():
    active_device = "cuda"
else:
    active_device = "cpu"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": active_device},
)

if os.path.exists("./chroma_store"):
    print("✅ Existing vector store found. Loading collection...")
    vectorstore = Chroma(
        persist_directory="./chroma_store",
        embedding_function=embedding_model,
        collection_name="doc-buddy",
    )
else:
    print("No documents indexed yet. Vector store initialized to None.")
    vectorstore = None
System_prompt = """
You are a precise document assistant.
Answer the user's question using ONLY the context provided below.
Rules:
- If the answer is not in the context, say exactly: "I don't have that information in the uploaded documents."
- Never use your general training knowledge to supplement the context.
- After your answer, add a 'Sources:' line citing the [Source N] labels you used.
- Keep answers concise and factual.
"""


def ask(question: str) -> tuple[str, str]:

    global vectorstore
    if vectorstore is None and os.path.exists("./chroma_store"):
        print("New database detected! Connecting retriever...")
        vectorstore = Chroma(
            persist_directory="./chroma_store",
            embedding_function=embedding_model,
            collection_name="doc-buddy",
        )
    if vectorstore is None or not os.path.exists("./chroma_store"):
        status = "No documents indexed yet"
        print(status)
        return (status, "")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    raw_chunks = retriever.invoke(question)
    for i, doc in enumerate(raw_chunks, 1):
        src = doc.metadata.get("source", "?")
        pg = doc.metadata.get("page", "?")
        print(f"  [{i}] {src} p.{pg}: {doc.page_content[:80]}…")

    context = "\n\n".join(
        [
            f"[Source {i}: {doc.metadata.get('source','?')}, page {doc.metadata.get('page','?')}]\n{doc.page_content}"
            for i, doc in enumerate(raw_chunks, 1)
        ]
    )
    context_display = "\n\n".join(
        [
            f"**[{doc.metadata.get('source', '?')} · Page {doc.metadata.get('page', '?')}]**\n"
            f"{doc.page_content[:200]}..."  # Truncated text preview
            for doc in raw_chunks
        ]
    )

    user_context = f"""
    Context : {context},
    Question : {question}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": System_prompt},
            {"role": "user", "content": user_context},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content

    return (answer, context_display)
