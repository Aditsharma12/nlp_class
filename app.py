from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import pickle
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# =========================
# LOAD MODEL + VECTORIZER
# =========================
session = ort.InferenceSession("model.onnx")
input_name  = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# =========================
# PREPROCESSING
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\bhowever\b|\balthough\b|\bthough\b", "but", text)
    return text


def apply_contrast_weighting(text):
    if "but" in text:
        parts = text.split("but")
        if len(parts) >= 2:
            before = parts[0]
            after = "but".join(parts[1:])
            return before + " " + (after + " " + after)  # 2x weight, matches training
    return text


# =========================
# ONNX PREDICTION
# =========================
def onnx_predict(text):
    # Transform raw text → TF-IDF dense features
    features = vectorizer.transform([text]).toarray().astype(np.float32)

    # Run through TF/ONNX model → softmax output shape (1, 2)
    outputs = session.run(output_names, {input_name: features})
    probs = outputs[0][0]  # [neg_prob, pos_prob]

    neg = float(probs[0])
    pos = float(probs[1])

    return pos, neg


# =========================
# VADER EXTRACTIONS
# =========================
def extract_words(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    pos_w = []
    neg_w = []
    for w in set(words):
        score = sia.polarity_scores(w)["compound"]
        if score >= 0.15:
            pos_w.append(w)
        elif score <= -0.15:
            neg_w.append(w)
    return {"positive": pos_w, "negative": neg_w}

def extract_portions(original_text):
    text_lower = original_text.lower()
    patterns = [r'\bbut\b', r'\bhowever\b', r'\balthough\b', r'\byet\b']
    
    for pat in patterns:
        match = re.search(pat, text_lower)
        if match:
            idx = match.start()
            part1 = original_text[:idx].strip()
            part2 = original_text[idx:].strip()
            
            s1 = sia.polarity_scores(part1)["compound"]
            s2 = sia.polarity_scores(part2)["compound"]
            
            if s1 >= s2:
                return {"positive": part1, "negative": part2}
            else:
                return {"positive": part2, "negative": part1}
    return {"positive": "", "negative": ""}


# =========================
# FINAL PREDICTION
# =========================
def predict_sentiment(text):
    try:
        original_text = text

        # Preprocess
        text = preprocess_text(text)
        weighted_text = apply_contrast_weighting(text)

        words = extract_words(original_text)
        portions = extract_portions(original_text)

        # ONNX DL prediction
        pos, neg = onnx_predict(weighted_text)

        confidence = max(pos, neg)
        sentiment = "Positive" if pos > neg else "Negative"

        # Detect Mixed
        if portions["positive"] and portions["negative"]:
            sentiment = "Mixed"
            confidence = (pos + neg) / 2.0  # balancing it out

        return {
            "input_text": original_text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": {
                "positive": round(pos, 4),
                "negative": round(neg, 4)
            },
            "words": words,
            "portions": portions,
            "analysis": {
                "certainty_level": (
                    "High" if confidence > 0.8 else
                    "Medium" if confidence > 0.6 else
                    "Low"
                ),
                "polarity_score": round(pos - neg, 4),
                "logic_applied": "TensorFlow Deep Learning + NLTK VADER Structural Analysis",
                "suggestion_for_llm": f"The sentiment is {sentiment.lower()} with {round(confidence*100,2)}% confidence."
            }
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No input"}), 400

    text = data["text"].strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    result = predict_sentiment(text)
    return jsonify(result)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)