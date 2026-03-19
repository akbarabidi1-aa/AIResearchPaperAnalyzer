import re

def summarize_large_text(text):
    """
    Extractive summarizer — no ML model needed, works instantly.
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    if not sentences:
        return "Could not extract summary from this document."

    # Word frequency scoring
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Normalize frequencies
    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_freq

    # Score each sentence
    scores = {}
    for i, sent in enumerate(sentences):
        score = 0
        for w in re.findall(r'\b[a-z]{4,}\b', sent.lower()):
            score += freq.get(w, 0)
        scores[i] = score

    # Pick top 5 sentences in original order
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    top = sorted(top)
    return ' '.join(sentences[i] for i in top)