import streamlit as st
from language_model_driver import train_model, train_model_word_level, predict_next_from_input

# Load models by default (without using Streamlit caching)
def load_default_models(train_file, max_order=5, debug=False):
    lm_char_model, vocab_char = train_model(train_file, max_order, debug)
    lm_word_model, vocab_word = train_model_word_level(train_file, max_order*2, debug)
    return (lm_char_model, vocab_char), (lm_word_model, vocab_word)

# Initialize the models when the app starts or reloads
(lm_char, vocab_char), (lm_word, vocab_word) = load_default_models("corpora/training-crowdsourcedaac.txt")

st.title('Interactive PPM Language Model Prediction')

st.write("Note all in memory. No fancy LLM involved. No requirements.txt. All training on the fly")

# Text input for user to type in their text
user_input = st.text_input("Type here:", key="user_input", on_change=st.rerun)
if 'n' not in st.session_state:
    st.session_state.n = 0  # Initialize once

if user_input:
    # Predictions
    #st.write(f"Debug: { st.session_state.n }")
    #st.write(f"Debug: {user_input}")
    st.session_state.n += 1
    predictions_char,_ = predict_next_from_input(lm_char, vocab_char, user_input, 5)
    predictions_word,_ = predict_next_from_input(lm_word, vocab_word, user_input, 5)

    # Debug output to see if predictions are updated
    #st.write(f"Debug: {predictions_char}")
    #st.write(f"Debug: {predictions_word}")

    st.write("Character Predictions:")
    col1, col2, col3, col4, col5 = st.columns(5)  # Adjust the number of columns based on your needs
    cols = [col1, col2, col3, col4, col5]
    for idx, char in enumerate(predictions_char):
        with cols[idx]:  # Use the column to display the button
            st.button(char, key=f"char_{idx}", on_click=lambda ch=char: st.session_state.update({'user_input': user_input + ch}))

    # Display word predictions horizontally
    st.write("Word Predictions:")
    col1, col2, col3, col4, col5 = st.columns(5)  # Adjust the number of columns based on your needs
    cols = [col1, col2, col3, col4, col5]
    for idx, word in enumerate(predictions_word):
        with cols[idx]:  # Use the column to display the button
            st.button(word, key=f"word_{idx}", on_click=lambda w=word: st.session_state.update({'user_input': user_input + w + " "}))