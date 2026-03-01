# ./examples/04_rag_pipeline.py
"""
RAG Pipeline Example
====================
Demonstrates vector storage and semantic search for retrieval-augmented generation.

Run: python examples/04_rag_pipeline.py
"""

import asyncio
from ollamatoolkit.tools.vector import VectorTools
from ollamatoolkit import SimpleAgent, ModelSelector


async def main():
    selector = ModelSelector()

    # Get embedding model
    embed_model = selector.get_best_embedding_model()
    if not embed_model:
        print("No embedding model found. Run: ollama pull nomic-embed-text")
        return

    print(f"Using embedding model: {embed_model}")

    # Create vector store
    vector_tools = VectorTools(
        embedding_model=embed_model,
        chunk_size=200,
        chunk_overlap=20,
    )

    # Example documents to ingest
    documents = [
        "Python is a high-level programming language known for readability.",
        "Machine learning uses algorithms to learn patterns from data.",
        "Neural networks are inspired by biological neural networks.",
        "PyTorch and TensorFlow are popular deep learning frameworks.",
        "Natural language processing enables computers to understand text.",
    ]

    # Ingest documents
    print("\n--- Ingesting Documents ---")
    for i, doc in enumerate(documents):
        await vector_tools.ingest_text(doc, metadata={"source": f"doc_{i}"})
        print(f"Ingested document {i + 1}")

    # Search for relevant documents
    print("\n--- Semantic Search ---")
    query = "What frameworks are used for deep learning?"
    results = await vector_tools.search(query, top_k=2)

    print(f"Query: {query}")
    print("Top results:")
    for i, result in enumerate(results):
        print(f"  {i + 1}. (score: {result['score']:.3f}) {result['text'][:80]}...")

    # Use in RAG pipeline
    print("\n--- RAG Response ---")

    chat_model = selector.get_best_chat_model()
    if chat_model:
        # Build context from search results
        context = "\n".join([r["text"] for r in results])

        agent = SimpleAgent(
            name="rag_agent",
            system_message=f"Answer based on this context:\n\n{context}",
            model_config={"model": f"ollama/{chat_model}"},
        )

        response = agent.run(query)
        print(f"Agent response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
