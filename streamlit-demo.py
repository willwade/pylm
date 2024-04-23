import streamlit as st
from language_model_driver import train_model, train_model_word_level, predict_next_from_input

# Load models by default
@st.cache_data(show_spinner=True)
def load_default_models(train_file, max_order=30, debug=False):
    lm_char_model, vocab_char = train_model(train_file, max_order, debug)
    lm_word_model, vocab_word = train_model_word_level(train_file, max_order*4, debug)
    return (lm_char_model, vocab_char), (lm_word_model, vocab_word)

(lm_char, vocab_char), (lm_word, vocab_word) = load_default_models("corpora/training-crowdsourcedaac.txt")

st.title('Interactive Language Model Prediction')

# Text input for user to type in their text
user_input = st.text_input("Type here:", key="user_input", on_change=st.rerun)
if 'n' not in st.session_state:
    st.session_state.n = 0  # Initialize once

if user_input:
    # Predictions
    st.write(f"Debug: { st.session_state.n }")
    st.write(f"Debug: {user_input}")
    st.session_state.n += 1
    predictions_char = predict_next_from_input(lm_char, vocab_char, user_input, 5)
    predictions_word = predict_next_from_input(lm_word, vocab_word, user_input, 5)

    # Debug output to see if predictions are updated
    st.write(f"Debug: {predictions_char}")
    st.write(f"Debug: {predictions_word}")

    # Display character predictions
    st.write("Character Predictions:")
    for idx, char in enumerate(predictions_char):
        st.button(char, key=f"char_{idx}", on_click=lambda ch=char: st.session_state.update({'user_input': user_input + ch}))

    # Display word predictions
    st.write("Word Predictions:")
    for idx, word in enumerate(predictions_word):
        st.button(word, key=f"word_{idx}", on_click=lambda w=word: st.session_state.update({'user_input': user_input + w + " "}))
