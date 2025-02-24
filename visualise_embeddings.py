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


### Variable to choose 2D or 3D plot
plot_dimension = '3d'    # '2d' for a 2D plot, '3d' for a 3D plot
###

# Reducing the dimensions with PCA
if plot_dimension == "2d":
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
elif plot_dimension == "3d":
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
else:
    raise ValueError("plot_dimension must be '2d' or '3d'.")

# Plotting the vectors
plt.figure(figsize=(10, 8))

if plot_dimension == "2d":
    # 2D scatter plot
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', edgecolors='k')
    # Annotation of 2D plot
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=10)

    plt.title("Word Embeddings Visualised with PCA (2D, GloVe 50)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)


elif plot_dimension == "3d":
    # 3D scatter plot
    ax = plt.subplot(111, projection='3d') 
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], c='blue', edgecolors='k')
    # Annotation of 3D plot
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=10) # type: ignore

    ax.set_title("Word Embeddings Visualised with PCA (3D, GloVe 50d)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3") # type: ignore

plt.show()