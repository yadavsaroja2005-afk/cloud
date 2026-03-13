# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.corpus import stopwords
import numpy as np
from numpy.linalg import norm

# Define training documents and query
train_set = ["The sky is blue.", "The sun is bright."]
test_set = ["The sun in the sky is bright."]

# Download stopwords
nltk.download('stopwords')
stopWords = stopwords.words('english')

# Initialize vectorizer and transformer
vectorizer = CountVectorizer(stop_words=stopWords)
transformer = TfidfTransformer()

# Convert documents into count vectors
trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()

# Display count vectors
print("Fit Vectorizer to train set")
print(trainVectorizerArray)

print("Transform Vectorizer to test set")
print(testVectorizerArray)

# Cosine similarity function
cx = lambda a, b: round(np.inner(a, b) / (norm(a) * norm(b)), 3)

# Print vectors and compute similarity
for vector in trainVectorizerArray:
    print("Train Vector:", vector)

for testV in testVectorizerArray:
    print("Test Vector:", testV)

    for vector in trainVectorizerArray:
        cosine = cx(vector, testV)
        print("Cosine Similarity:", cosine)

# Convert count vectors to TF-IDF
transformer.fit(trainVectorizerArray)
print("\nTF-IDF for Training Documents")
print(transformer.transform(trainVectorizerArray).toarray())

transformer.fit(testVectorizerArray)
print("\nTF-IDF for Query")
tfidf = transformer.transform(testVectorizerArray)
print(tfidf.todense())
