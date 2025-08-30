import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# Download NLTK resources
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ----------------------------
# Load model
# ----------------------------
model_path = "results/best"  # <-- adjust if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----------------------------
# Label mapping
# ----------------------------
label_map = {0: "Advertisement", 1: "Irrelevant Content", 2: "Rant without visiting", 3: "None"}

# ----------------------------
# Preprocessing
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_texts(texts):
    """
    Lowercase, remove punctuation, remove stopwords, lemmatize.
    """
    processed = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Tokenize and remove stopwords
        tokens = [w for w in text.split() if w not in stop_words]
        # Lemmatize
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        processed.append(" ".join(tokens))
    return processed

# ----------------------------
# Custom prediction with thresholds
# ----------------------------
# Example thresholds per class: adjust as needed
default_thresholds = [0.5316590285922207, 0.5266510967355174, 0.5370496628602927]  # for classes 0,1,2
none_class = 3

def predict_with_thresholds(texts, thresholds=default_thresholds, batch_size=32):
    """
    Predict class labels with per-class thresholds and margin rule.
    """
    model.eval()
    model.to(device)
    preds = []

    texts = preprocess_texts(texts)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for p in probs:
                top_idx = int(np.argmax(p))
                top_prob = p[top_idx]

                if top_idx != none_class and top_prob >= thresholds[top_idx]:
                    preds.append(top_idx)
                else:
                    preds.append(none_class)
    return preds

# ----------------------------
# Wrapper for Streamlit
# ----------------------------
def predict_text(texts):
    """
    Accepts list of strings, returns predicted labels.
    """
    if isinstance(texts, str):
        texts = [texts]
    pred_indices = predict_with_thresholds(texts)
    return [label_map[idx] for idx in pred_indices]