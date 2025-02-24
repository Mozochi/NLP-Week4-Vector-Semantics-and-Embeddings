import numpy as np
from gensim.downloader import load
from scipy.spatial.distance import cosine

# Loading the model
print("Loading GloVe model...")
glove_model = load('glove-wiki-gigaword-50') # 50d vector
print("GloVe model loaded.")


def compute_cosine_similarity(word1, word2):
    try:
        # Getting the vectors for both words
        vec1 = glove_model[word1]
        vec2 = glove_model[word2]

        # Computing the cosine similarity (Cosine similarity = 1 - Cosine distance)
        similarity = 1 - cosine(vec1, vec2)
        return similarity
    except KeyError as e:
        print(f"Error: {e} not in vocabulary.")

# Test word pairs
word_pairs = [
    ("computer", "software"),
    ("king", "queen"),
    ("dog", "cat"),
    ("cold", "hot"),
    ("cat", "computer")
]


# Computing and displaying the cosine similarities 
for word1, word2 in word_pairs:
    similarity = compute_cosine_similarity(word1, word2)
    print(f"Similarity between {word1} and {word2}: {similarity:.3f}")