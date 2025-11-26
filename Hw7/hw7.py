import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# ------------------------------
# Streamlit page configuration
# ------------------------------
st.set_page_config(page_title="IMDB Movie Review Classifier by Sandtroe")
st.title("IMDB Movie Review Classifier by Sandtroe")

# ------------------------------
# Load model and IMDB metadata
# ------------------------------
max_features = 10_000
max_len = 500

# Load the trained model (make sure the .keras file or .h5 is in the same folder)
model = load_model("simple_rnn_imdb.keras")
word_index = imdb.get_word_index()

# ------------------------------
# Text preprocessing function
# ------------------------------
def tokenize(text: str):
    words = text.lower().split()
    encoded = []
    for w in words:
        idx = word_index.get(w, 2) + 3  # +3 because IMDB word index starts from 3
        # Fix: ensure index stays inside valid range [0, max_features)
        if idx >= max_features:
            idx = 2  # unknown token
        encoded.append(idx)
    return sequence.pad_sequences([encoded], maxlen=max_len)

# ------------------------------
# User input
# ------------------------------
st.write("Enter up to 5 movie reviews (one per box), then click **Classify**.")

reviews = [st.text_area(f"Review {i+1}", height=120, key=f"review_{i}") for i in range(5)]

# ------------------------------
# Classification
# ------------------------------
if st.button("Classify"):
    for i, txt in enumerate(reviews, start=1):
        if not txt.strip():
            st.write(f"Review {i}: (empty)")
            continue

        try:
            x = tokenize(txt)
            p = float(model.predict(x, verbose=0)[0][0])
            label = "Positive" if p >= 0.5 else "Negative"
            st.write(f"**Review {i}: {label} (p={p:.3f})**")
        except Exception as e:
            st.error(f"Error processing Review {i}: {e}")

# ------------------------------
# Footer note
# ------------------------------
st.markdown("---")
st.caption("IMDB Movie Review Classifier by Sandtroe — built with a SimpleRNN model trained in TensorFlow/Keras.")
