import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Only need to download these once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
    # 1. Tokenize and Lowercase
    tokens = word_tokenize(text.lower())
    
    # 2. Get stop words and punctuation list
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    # 3. Filter out stop words AND punctuation AND non-alphabetic tokens
    # (The .isalpha() check is great for removing numbers and symbols)
    filtered = [w for w in tokens if w not in stop_words 
                and w not in punctuation 
                and w.isalpha()]
    
    return filtered

# Example usage with your PDF text:
# cleaned_tokens = clean_text(clean_text_from_pdf)
# print(cleaned_tokens[:20])