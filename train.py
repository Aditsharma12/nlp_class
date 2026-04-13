import sys
import os
import pickle
import shutil
import subprocess
import numpy as np
import pandas as pd
import re

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tf2onnx


# =========================
# TEXT PREPROCESSING
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\bhowever\b", "but", text)
    text = re.sub(r"\balthough\b", "but", text)
    text = re.sub(r"\bthough\b", "but", text)
    return text


def apply_contrast_weighting(text):
    """Give more importance to the part after 'but'"""
    if "but" in text:
        parts = text.split("but")
        if len(parts) >= 2:
            before = parts[0]
            after = "but".join(parts[1:])
            return before + " " + (after + " " + after)  # 2x weight
    return text


# =========================
# MAIN TRAIN FUNCTION
# =========================
def train_model():
    print("1. Loading dataset...")

    dataset = load_dataset("amazon_polarity", split="train[:50000]")
    df = pd.DataFrame(dataset)

    df["text"] = df["title"].fillna("") + " " + df["content"]
    # label stays as 0 (Negative) / 1 (Positive)

    df = df[(df["text"].str.len() > 20) & (df["text"].str.len() < 1500)]

    df["text"] = df["text"].apply(preprocess_text)
    df["text"] = df["text"].apply(apply_contrast_weighting)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    # =========================
    # TF-IDF VECTORIZER
    # =========================
    print("2. Fitting TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train).toarray().astype(np.float32)
    X_test_tfidf  = vectorizer.transform(X_test).toarray().astype(np.float32)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("   [OK] Vectorizer saved as vectorizer.pkl")

    # =========================
    # TENSORFLOW / KERAS MODEL
    # =========================
    print("3. Training TensorFlow (Keras) model...")

    n_features = X_train_tfidf.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(n_features,), name="input"),

        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.4),

        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(2, activation="softmax", name="output"),  # [neg_prob, pos_prob]
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(
        X_train_tfidf,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    # =========================
    # EVALUATION
    # =========================
    train_preds = np.argmax(model.predict(X_train_tfidf, verbose=0), axis=1)
    test_preds  = np.argmax(model.predict(X_test_tfidf,  verbose=0), axis=1)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)

    print(f"\nTrain Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc  * 100:.2f}%\n")

    # =========================
    # EXPORT TO ONNX
    # =========================
    print("4. Exporting model to ONNX...")

    # Save as TF SavedModel first (tf2onnx.from_keras is broken with Keras 3.x)
    saved_model_path = "saved_model_temp"
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)
    tf.saved_model.save(model, saved_model_path)

    # Convert SavedModel -> ONNX via CLI (most reliable cross-version method)
    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", saved_model_path,
            "--output", "model.onnx",
            "--opset", "13",
        ],
        capture_output=True,
        text=True,
    )

    shutil.rmtree(saved_model_path, ignore_errors=True)

    if result.returncode != 0:
        print("ONNX conversion failed:")
        print(result.stderr)
        sys.exit(1)

    print("[OK] Model saved as model.onnx")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train_model()