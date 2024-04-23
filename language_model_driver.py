# python language_model_driver.py 30 path_to_train.txt path_to_test.txt

from string import punctuation
import sys
from vocabulary import Vocabulary
from ppm_language_model import PPMLanguageModel, Context
import math

# IF You are using any of the plotting or other functions you need these
plotting = False

if plotting:
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import re


def train_model(train_file, max_order, debug=False):
    with open(train_file, 'r', encoding='utf-8') as file:
        contents = file.read()

    vocab = Vocabulary()
    for char in set(contents):
        vocab.add_item(char)

    lm = PPMLanguageModel(vocab, max_order, debug=debug)
    context = Context(lm.root, 0)
    for char in contents:
        symbol_id = vocab.get_id_or_oov(char)  # Ensuring OOV handling
        lm.add_symbol_and_update(context, symbol_id)

    return lm, vocab


def test_model(lm, vocab, test_file):
    with open(test_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total_log_prob = 0
    num_symbols = 0
    for line in lines:
        if line.strip():
            context = Context(lm.root, 0)
            for char in line.strip():
                symbol = vocab.get_id_or_oov(char)
                probs = lm.get_probs(context)
                # Handling OOV and checking bounds
                prob = probs[symbol] if symbol < len(probs) else 0
                if prob > 0:
                    total_log_prob += math.log10(prob)
                    num_symbols += 1
                lm.add_symbol_to_context(context, symbol)

    entropy = -(total_log_prob / num_symbols) / \
        math.log10(2) if num_symbols > 0 else 0
    perplexity = 10 ** (-total_log_prob /
                        num_symbols) if num_symbols > 0 else float('inf')
    print(
        f"Results: numSymbols = {num_symbols}, ppl = {perplexity}, entropy = {entropy} bits/char")


def tokenize(text):
    # Replace characters that are repeated three or more times with a single character
    text = re.sub(
        r'[{}]+'.format(re.escape(punctuation.replace('?!\'', ''))), ' ', text)

    # Remove specific characters
    text = re.sub(r'[?!\'"]+', '', text)

    # Condense all whitespace to a single space
    text = re.sub(r'\s+', ' ', text)

    # Trim leading and trailing whitespaces
    text = text.strip()

    # Split the text into tokens
    tokens = text.split()
    return tokens


def train_model_word_level(train_file, max_order, debug=False):
    with open(train_file, 'r', encoding='utf-8') as file:
        contents = file.read()

    words = tokenize(contents)
    vocab = Vocabulary()
    for word in set(words):
        vocab.add_item(word)

    lm = PPMLanguageModel(vocab, max_order, debug=debug)
    context = Context(lm.root, 0)
    for word in words:
        word_id = vocab.get_id_or_oov(word)
        lm.add_symbol_and_update(context, word_id)

    return lm, vocab

# ... other parts of your code ...


def get_context_node(lm, context):
    current_node = lm.root
    print(f"Starting search for context '{context}'")
    for char in context:
        # Ensure to work with character IDs
        char_id = lm.vocab.get_id_or_oov(char)
        if char_id in current_node.children:
            current_node = current_node.children[char_id]
        else:
            print(f"Context '{context}' not found at character '{char}'")
            return None
    # Return a Context object instead of Node
    return Context(current_node, len(context))


def predict_next_from_input(lm, vocab, input_text, num_predictions=3):
    context = lm.create_context()
    prob_symbol_pairs = []

    try:
        for char in input_text:
            char_id = vocab.get_id_or_oov(char)
            lm.add_symbol_to_context(context, char_id)
            if lm.debug:
                print(
                    f"Adding '{char}' to context, ID: {char_id}, Current Head ID: {id(context.head)}, Order: {context.order}")

        if lm.debug:
            print("Attempting to predict next symbols...")

        probs = lm.get_probs(context)
        prob_symbol_pairs = lm.get_probs_with_symbols(context)

        # Sort by probability and get the top predictions
        top_prediction_ids = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[
            :num_predictions]
        if lm.debug:
            print(f"Probabilities: {probs}")
            print(f"Top prediction IDs: {top_prediction_ids}")
        predicted_chars = [vocab.get_item_by_id(
            index) if index != vocab.oov_index else '<OOV>' for index, _ in top_prediction_ids]

        if lm.debug:
            print("Prediction IDs retrieved, processing output...")
            print(
                f"Final Context State before Prediction: Head at {id(context.head)}, Order: {context.order}")

        return predicted_chars, prob_symbol_pairs
    except Exception as e:
        print(f"Error during context update: {str(e)}")
        # Return an empty list and prob_symbol_pairs if an error occurs
        return [], prob_symbol_pairs


# Helper function to plot the probabilities with corresponding symbols


def plot_probabilities(prob_symbol_pairs):
    # Unzip the pairs into two lists
    prob_symbol_pairs_sorted = sorted(
        prob_symbol_pairs, key=lambda x: x[1], reverse=True)

    # Extract just the probabilities after sorting
    probabilities = [prob for symbol, prob in prob_symbol_pairs_sorted]

    # Rank the probabilities (highest probability gets rank 1)
    ranks = np.arange(1, len(probabilities) + 1)

    # Convert ranks and probabilities to a logarithmic scale
    log_ranks = np.log(ranks)
    log_probabilities = np.log(probabilities)

    # Plotting on a log-log scale
    plt.figure(figsize=(10, 6))
    plt.plot(log_ranks, log_probabilities, marker='o', linestyle='',
             markersize=5, label='Log-Log Distribution')

    plt.xlabel('Log of Rank')
    plt.ylabel('Log of Probability')
    plt.title('Log-Log Probability Distribution of Symbols')
    plt.legend()
    plt.grid(True)
    plt.show()


'''
    So this next chunk is really code to give you a very pretty graph of the trie
    I'm going to comment out the imports. Sometimes installing graphviz isnt fun

'''


def build_graph_iterative(root, vocab, max_depth=20):
    graph = nx.DiGraph()
    stack = [(root, "Root", "", 0)]  # node, parent ID, symbol, current depth

    while stack:
        node, parent_id, symbol, depth = stack.pop()

        if depth >= max_depth:
            continue

        symbol_label = f"<space>" if symbol == " " else symbol
        node_id = f"{id(node)}_{symbol if symbol else 'Root'}"

        graph.add_node(node_id, label=f"{symbol_label}\nCount: {node.count}")
        if parent_id != "Root":
            graph.add_edge(parent_id, node_id)

        for child_symbol_id, child in node.children.items():
            child_symbol = vocab.get_item_by_id(child_symbol_id)
            stack.append((child, node_id, child_symbol, depth + 1))

    return graph


def draw_graph(graph):
    # Use a default label if 'label' key is not found
    labels = {n: data.get('label', 'No Label')
              for n, data in graph.nodes(data=True)}

    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, labels=labels, with_labels=True,
            node_size=2000, node_color='lightblue')
    plt.show()

# Assuming 'lm' is your language model and 'vocab' is your vocabulary object

# 1. Print context frequencies


def print_context_frequencies(lm, context):
    """Print frequencies of following characters for a given context."""
    current_node = lm.root
    for char in context:
        if not current_node:
            print(f"Context '{context}' not found at character '{char}'")
            return
        char_id = vocab.get_id_or_oov(char)
        current_node = current_node.find_child_with_symbol(char_id)
    if current_node:
        print(f"Frequencies after context '{context}':")
        for symbol_id, child in current_node.children.items():
            char = vocab.get_item_by_id(symbol_id)
            print(f"'{char}': {child.count}")
    else:
        print(f"Context node for '{context}' not found.")


def print_contexts(node, vocab, current_context='', depth=0, max_depth=5):
    if depth > max_depth:  # Limit the depth to prevent too much output
        return
    for symbol_id, child_node in node.children.items():
        symbol = vocab.get_item_by_id(symbol_id)  # Get symbol from vocab
        new_context = current_context + symbol
        print(f"{' ' * depth}{new_context}")
        print_contexts(child_node, vocab, new_context, depth + 1, max_depth)


# 3. Vocabulary and mapping checks
def vocab_and_mapping_checks(vocab, training_text):
    # Verify vocabulary size
    expected_vocab_size = len(set(training_text))
    actual_vocab_size = vocab.size()
    print(
        f"Expected vocab size: {expected_vocab_size}, Actual: {actual_vocab_size}")

    # Check unique character IDs
    unique_ids = set(vocab.get_id_or_oov(char) for char in training_text)
    if len(unique_ids) != actual_vocab_size:
        print(f"Mismatch in unique character IDs and vocabulary size.")

    # Check for character representation in the vocabulary
    missing_chars = {char for char in training_text if vocab.get_id_or_oov(
        char) == vocab.oov_index}
    for char in missing_chars:
        print(f"Character '{char}' not found in vocabulary.")


def check_prob_distribution_changes(lm, vocab, context_examples):
    for context_string in context_examples:
        # This should return a Context now
        context = get_context_node(lm, context_string)
        if context:
            probs = lm.get_probs(context)
            print(f"Probabilities for context '{context_string}': {probs}")
        else:
            print(f"Context '{context_string}' not found.")


def check_vocab(vocab):
    print(f"Vocabulary size: {vocab.size()}")
    print("Sample items in vocabulary:")
    for i in range(min(vocab.size(), 20)):  # Print first 20 items for brevity
        print(f"ID {i}: {vocab.get_item_by_id(i)}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py max_order train_file test_file debug")
        sys.exit(1)
    max_order, train_file, test_file = int(
        sys.argv[1]), sys.argv[2], sys.argv[3]
    lm, vocab = train_model(train_file, max_order, debug=True)
    
    
    plotting=True

    test_model(lm, vocab, test_file)
    print('Space character ID:', vocab.get_id_or_oov(' '))
    print('Is space in vocabulary:', ' ' in vocab.items_to_index)

    # Usage examples
    context_examples = ['the', 'and', 'a']
    print_context_frequencies(lm, 'the')  # Example context
    print_contexts(lm.root, vocab)

    # Debugging and testing the function
    context_node = get_context_node(lm, 'the')
    if context_node:
        print(f"Context node for 'the' found.")
    else:
        print(f"Context node for 'the' not found.")

    check_prob_distribution_changes(lm, vocab, context_examples)
    vocab_and_mapping_checks(vocab, 'Sample training text to check.')

    input_text = "Wh"
    num_predictions = 5
    # For character-level prediction
    lm.debug = True
    predicted_chars, prob_symbol_pairs = predict_next_from_input(
        lm, vocab, input_text, num_predictions)
    print(f"Top 5 character predictions for 'he': {predicted_chars}")
    # plot_probabilities(prob_symbol_pairs)

    # draw_graph(build_graph_iterative(lm.root, vocab, 20))
    # For word-level predictions
    lm_word, vocab_word = train_model_word_level(
        train_file, max_order, debug=False)  # Use the word-level training function
    input_text = "wh"
    num_predictions = 5
    # For character-level predictions
    predicted_words, prob_symbol_pairs = predict_next_from_input(
        lm_word, vocab_word, input_text, num_predictions)
    print(f"Top 5 word predictions for 'What': {predicted_words}")
    plot_probabilities(prob_symbol_pairs)
