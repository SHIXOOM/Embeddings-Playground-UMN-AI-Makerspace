from gensim.models import KeyedVectors


def load_static_embeddings(file_path, binary=True, no_header=False) -> KeyedVectors:
    try:
        # Load pre-trained embeddings
        embeddings = KeyedVectors.load_word2vec_format(
            file_path, binary=binary, no_header=no_header
        )
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


# Example
if __name__ == "__main__":
    embeddings: KeyedVectors = load_static_embeddings(
        "embeddings/dolma_300_2024_1.2M.100_combined.txt", binary=False, no_header=True
    )
    print("\nBefore similarity search\n")
    print(
        embeddings.most_similar(
            positive=["capital", "russia"], negative=["country"], topn=5
        )
    )
