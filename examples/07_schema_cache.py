# ./examples/07_schema_cache.py
"""
Schema & Cache Tools Example
============================
Demonstrates JSON schema validation and response caching.

Run: python examples/07_schema_cache.py
"""

from ollamatoolkit.tools.schema import SchemaTools
from ollamatoolkit.tools.cache import CacheTools


def main():
    # =========================================================================
    # Part 1: Schema Tools
    # =========================================================================
    print("=" * 60)
    print("SCHEMA TOOLS DEMO")
    print("=" * 60)

    schema_tools = SchemaTools()

    # Define a schema for structured output
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string"},
            "roles": {
                "type": "array",
                "items": {"type": "string", "enum": ["admin", "user", "guest"]},
            },
        },
        "required": ["name", "age"],
    }

    # Validate data
    print("\n--- Schema Validation ---")

    valid_data = {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
        "roles": ["admin"],
    }
    result = schema_tools.validate(valid_data, person_schema)
    print(f"Valid data: {result.valid}")  # True

    invalid_data = {"name": "", "age": -5}
    result = schema_tools.validate(invalid_data, person_schema)
    print(f"Invalid data: {result.valid}")  # False
    print(f"Errors: {result.errors}")

    # Generate sample data
    print("\n--- Generate Sample ---")
    sample = schema_tools.generate_sample(person_schema)
    print(f"Sample: {sample}")

    # Infer schema from example
    print("\n--- Infer Schema from Example ---")
    example = {"product": "Widget", "price": 9.99, "in_stock": True}
    inferred = schema_tools.from_example(example)
    print(f"Inferred: {inferred}")

    # Generate LLM prompt
    print("\n--- Schema to Prompt ---")
    prompt = schema_tools.to_prompt(person_schema)
    print(prompt)

    # =========================================================================
    # Part 2: Cache Tools
    # =========================================================================
    print("\n" + "=" * 60)
    print("CACHE TOOLS DEMO")
    print("=" * 60)

    cache = CacheTools(cache_dir="./cache_demo", default_ttl=3600)

    # Cache a response
    print("\n--- Response Caching ---")

    prompt = "What is the capital of France?"
    model = "llama3.1"

    # First call - cache miss
    cached = cache.get_response(prompt, model)
    print(f"First lookup (miss): {cached}")

    # Simulate LLM response and cache it
    response = "The capital of France is Paris."
    cache.cache_response(prompt, model, response, temperature=0.0)
    print("Cached response")

    # Second call - cache hit
    cached = cache.get_response(prompt, model)
    print(f"Second lookup (hit): {cached}")

    # Embedding caching
    print("\n--- Embedding Caching ---")
    text = "Hello world"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding

    cache.cache_embedding(text, "nomic-embed-text", embedding)
    cached_embedding = cache.get_embedding(text, "nomic-embed-text")
    print(f"Cached embedding: {cached_embedding[:3]}...")

    # Statistics
    print("\n--- Cache Statistics ---")
    stats = cache.stats()
    print(f"Total entries: {stats.total_entries}")
    print(f"Hits: {stats.hits}")
    print(f"Misses: {stats.misses}")
    print(f"Hit rate: {stats.hit_rate:.1f}%")

    # Cleanup
    cache.clear()
    print("\nCache cleared!")


if __name__ == "__main__":
    main()
