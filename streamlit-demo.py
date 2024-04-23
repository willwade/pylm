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

st.write("Note all in memory. No fancy LLM involved. No requirements.txt. All training on the fly.Ignore the widget warning. No idea how to get rid of that!")

# Define the handler for updating the user input
def update_input(addition):
    st.session_state.user_input += addition

# Initialize or retrieve the current user input from session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Display the text input field, bound directly to session state
user_input = st.text_input("Type here:", value=st.session_state.user_input, key="user_input")

if user_input:
    # Fetch predictions based on the current input
    predictions_char,_ = predict_next_from_input(lm_char, vocab_char, user_input, 5)
    predictions_word,_ = predict_next_from_input(lm_word, vocab_word, user_input, 5)

    # Display character predictions in a horizontal layout
    st.write("Character Predictions:")
    cols = st.columns(len(predictions_char))  # Dynamically adjust based on the number of predictions
    for idx, char in enumerate(predictions_char):
        with cols[idx]:
            st.button(char, key=f"char_{idx}", on_click=update_input, args=(char,))

    # Display word predictions in a horizontal layout
    st.write("Word Predictions:")
    cols = st.columns(len(predictions_word))
    for idx, word in enumerate(predictions_word):
        with cols[idx]:
            st.button(word, key=f"word_{idx}", on_click=update_input, args=(word + " ",))
