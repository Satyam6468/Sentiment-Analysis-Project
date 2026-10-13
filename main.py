import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences

# Load model and tokenizer
model = load_model("Sentimentanalysis.h5")
with open("tokenizer.pkl", "rb") as file:
  tokenizer = pickle.load(file)

classes = ["Negative", "Neutral", "Positive"]

st.title("Sentiment Analysis")

# User input
user_input = st.text_input("Enter a comment:")

if st.button("Predict"):
  if user_input:
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=model.input_shape[1], padding="pre")

    # Predict
    pred = model.predict(padded)
    result = classes[np.argmax(pred)]

    st.write(f"**Sentiment:** {result}")
  else:
    st.warning("Please type something first.")