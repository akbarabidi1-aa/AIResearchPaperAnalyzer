import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # transform takes a list of documents
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Get the feature names and their scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # Create a list of (word, score) tuples and sort them
    keyword_series = dict(zip(feature_names, scores))
    sorted_keywords = sorted(keyword_series.items(), key=lambda x: x[1], reverse=True)
    
    # Return just the top N words
    return [word for word, score in sorted_keywords[:top_n]]

# Example Output: ['neural', 'backpropagation', 'optimization', 'weights'...]