import numpy as np
from gensim.downloader import load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Loading the model
print("Loading GloVe model...")
glove_model = load('glove-wiki-gigaword-50') # 50d vector
print("GloVe model loaded.")

# Words to visualise (a mix of related and unrelated terms)
words = [
    "cat", "dog", "bird", "hamster",        # Pets
    "computer", "software", "programming",  # Tech
    "pizza", "pasta", "burger", "salad"     # Food

]


# Getting vectors for the words
vectors = np.array([glove_model[word] for word in words] ) # type: ignore

# Reducing the dimensions to 2D with PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Plotting the vectors
plt.figure(figsize=(10, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', edgecolors='k')

# Annotation of plot
for i, word in enumerate(words):
    plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=10)

plt.title("Word Embeddings Visualiced with PCA (GloVe 50)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()