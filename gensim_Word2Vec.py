from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
import re

news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = news.data # type: ignore # 

def preprocess_text(text):
    # Convert all text to lowercase and removing special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenizing the text
    tokens = word_tokenize(text)

    return tokens

# Processing the text
tokenized_text = [preprocess_text(text) for text in texts[:100]] # Limiting to 100 texts to reduce processing time

# Training the WordVec model
model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4, sg=0)  # Input data, Size of word vector, context window size, minimum word frequency, number of cpu threads, 0 for CBOW, 1 for skip-gram

# model.save("word2vec.model")  # Saving the model
# print("Vocabulary:", list(model.wv.key_to_index.keys())) # Debugging/Finding the vocabulary

vector = model.wv['beginning']  
print("Vector for beginning: ", vector)  

similar_to_computers = model.wv.most_similar('beginning')
print("Words similar to beginning: ", similar_to_computers)

similarity = model.wv.similarity('sunday', 'friday')
print("Similarity between sunday and friday: ", similarity) 