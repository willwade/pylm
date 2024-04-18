import sys
import string
from collections import defaultdict

# Assuming nltk and datasets are installed; if not, you might need to install them:
# pip install nltk datasets

import nltk
from datasets import load_dataset

# Set recursion limit very high for deep recursion used by the trie (if necessary)
sys.setrecursionlimit(1000000)

# Define the Vocabulary class
class Vocabulary:
    def __init__(self,mode='character', debug=False):
        self.symbols = {}
        self.index_to_symbol = []
        self.root_symbol = "<R>"
        self.oov_symbol = "<OOV>"
        self.debug=debug
        self.add_symbol(self.root_symbol)  # Ensure root symbol is at index 0
        self.add_symbol(self.oov_symbol)  # Ensure OOV symbol is at index 1

    def add_symbol(self, symbol):
        if symbol not in self.symbols:
            index = len(self.index_to_symbol)
            self.symbols[symbol] = index
            self.index_to_symbol.append(symbol)
            if self.debug: 
                print(f"Symbol '{symbol}' added at index {index}")
        else:
            index = self.symbols[symbol]
            if self.debug: 
                print(f"Symbol '{symbol}' already exists at index {index}")
        if self.debug: 
            print(f"Current mapping: {self.index_to_symbol}")
        return index



    def get_symbol_id_or_oov(self, symbol):
        return self.symbols.get(symbol, self.symbols[self.oov_symbol])

    def get_symbol_by_id(self, index):
        try:
            symbol = self.index_to_symbol[index]
            if self.debug: 
                print(f"Successfully retrieved symbol '{symbol}' for index {index}")
            return symbol
        except IndexError:
            if self.debug: 
                print(f"Index {index} out of range, returning OOV symbol.")
            return self.oov_symbol  # Return OOV symbol if index is out of range


# Define the Context and Node classes
class Context:
    def __init__(self, order, mode='character',debug=False):
        self.entries = []
        self.order = order
        self.mode = mode  # 'character' or 'word'
        self.debug = debug
        
    def update_context(self, new_entry):
        """Adds a new entry to the context, maintaining the correct length."""
        self.entries.append(new_entry)
        while len(self.entries) > self.order:
            self.entries.pop(0)
        if self.debug: 
            print("Updated context entries:", self.entries)  # Debug statement

    def get_current_head(self):
        """Assuming that the last entry is considered the current head in the context."""
        return self.entries[-1] if self.entries else None

    def get_context(self):
        """Returns the full current context as a list of entries."""
        return self.entries
        
    def __str__(self):
        """Provides a string representation to track context's state for debugging."""
        return ' -> '.join(str(entry) for entry in self.entries)


class Node:
    """ 
    Node in a search tree, which is implemented as a suffix trie that represents
    every suffix of a sequence used during its construction. Please see
     [1] Moffat, Alistair (1990): "Implementing the PPM data compression
         scheme", IEEE Transactions on Communications, vol. 38, no. 11, pp.
         1917--1921.
     [2] Esko Ukknonen (1995): "On-line construction of suffix trees",
         Algorithmica, volume 14, pp. 249--260, Springer, 1995.
     [3] Kennington, C. (2011): "Application of Suffix Trees as an
         Implementation Technique for Varied-Length N-gram Language Models",
         MSc. Thesis, Saarland University.
    """
    def __init__(self, symbol=None):
        """
        Node in the backoff structure, also known as "vine" structure (see [1]
        above) and "suffix link" (see [2] above). The backoff for the given node
        points at the node representing the shorter context. For example, if the
        current node in the trie represents string "AA" (corresponding to the
        branch "[R] -> [A] -> [*A*]" in the trie, where [R] stands for root),
        then its backoff points at the node "A" (represented by "[R] ->
        [*A*]"). In this case both nodes are in the same branch but they don't
        need to be. For example, for the node "B" in the trie path for the string
        "AB" ("[R] -> [A] -> [*B*]") the backoff points at the child node of a
        different path "[R] -> [*B*]". 
        """
        self.child = None  # Leftmost child node for the current node
        self.next = None   # Next node
        self.backoff = None  # Node in the backoff structure   
        self.count = 1     # Frequency count for this node
        self.symbol = symbol  # Symbol that this node stores

    def add_child(self, symbol):
        if symbol not in self.children:
            self.children[symbol] = Node()
        return self.children[symbol]

    def find_child_with_symbol(self, symbol):
        """
        Finds a child node of the current node with a specified symbol.
        
        Args:
            symbol (int): Integer symbol to search for.
            
        Returns:
            Node: Node with the specified symbol, or None if not found.
        """    
        current = self.child
        while current is not None:
            if current.symbol == symbol:
                return current
            current = current.next
        return None
    
    def find_or_create_child(self, symbol):
        # First try to find the node
        node = self.find_child_with_symbol(symbol)
        if node is None:
            # Node doesn't exist, so create a new one
            node = Node(symbol)
            # Insert it as the first child for simplicity (or maintain sorted order if necessary)
            node.next = self.child
            self.child = node
        return node
            
    def total_children_counts(self, exclusion_mask=None):
        """
        Total number of observations for all the children of this node.
        This counts all the events observed in this context.
    
        Note: This API is used at inference time. A possible alternative that will
        speed up the inference is to store the number of children in each node as
        originally proposed by Moffat for PPMB in
          Moffat, Alistair (1990): "Implementing the PPM data compression scheme",
          IEEE Transactions on Communications, vol. 38, no. 11, pp. 1917--1921.
        This however will increase the memory use of the algorithm which is already
        quite substantial.
    
        Args:
            exclusion_mask (List[bool], optional): Boolean exclusion mask for all the symbols.
                Can be None, in which case no exclusion happens.
    
        Returns:
            int: Total number of observations under this node.
        """
        count = 0
        node = self.child
        while node:
            if exclusion_mask is None or (node.symbol < len(exclusion_mask) and not exclusion_mask[node.symbol]):
                count += node.count
            node = node.next
        return count

    def num_distinct_symbols(self):
        """
        Returns the number of distinct symbols (children) for this node.
        """
        distinct_count = 0
        current = self.child
        while current:
            distinct_count += 1
            current = current.next
        return distinct_count
        
    def iterate_children(self):
        """Yield each child node starting from the first child."""
        current = self.child
        while current:
            yield current
            current = current.next

    def __str__(self):
        return f"Node(symbol={self.symbol}, count={self.count})"

    def __repr__(self):
        return self.__str__()

# Define the PPMLanguageModel class
class PPMLanguageModel:
    def __init__(self, vocabulary, max_order, debug=False):
        self.vocab = vocabulary
        self.max_order = max_order
        self.root = Node()
        self.debug = debug

    def add_symbol_and_update(self, context, symbol):
        if self.debug: 
            print(f"Adding symbol: {symbol} to context")  # Symbol here should be an ID, not a character
        context.update_context(symbol)
        current_context = context.get_context()  # This should be a list of integers (symbol IDs)
        if self.debug: 
            print(f"Current path before adding new symbol: {' -> '.join(str(self.vocab.get_symbol_by_id(sid)) for sid in current_context)}")
    
        current_node = self.root
        for symbol_id in current_context:
            current_node = current_node.find_or_create_child(symbol_id)
            if current_node is None:
                if self.debug: 
                    print("Failed to find or create a node for symbol ID:", symbol_id)
                return
    
        current_node.count += 1
        if self.debug: 
            print(f"Updated path after adding new symbol: {self.vocab.get_symbol_by_id(current_node.symbol)}")
            self.print_trie()

    def add_symbol_to_context(self, context, symbol_id):
            """Adds a symbol to the context but does not update the trie."""
            context.update_context(symbol_id)  # Assumes update_context method properly manages the context size and entries

    def predict_next_indices(self, context, num_predictions=1):
        """Predict the indices of the next possible symbols."""
        current_node = self.find_node_by_context(context)
        if not current_node:
            print("No current node found for context:", context)
            return []

        # Get children and sort them by count in descending order
        predictions = []
        child = current_node.child
        while child:
            predictions.append((child.symbol, child.count))
            child = child.next

        predictions.sort(key=lambda x: x[1], reverse=True)  # Sort by count, descending
        top_predictions = [pred[0] for pred in predictions[:num_predictions]]

        return top_predictions

    def find_node_by_context(self, context):
        """Finds the node in the trie corresponding to the given context (list of symbol IDs)."""
        current_node = self.root
        for symbol_id in context:
            current_node = current_node.find_child_with_symbol(symbol_id)
            if not current_node:
                return None  # Context not found
        return current_node    

    def print_trie(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = " " * depth
        print(f"{indent}{node}")
        child = node.child
        while child:
            self.print_trie(child, depth + 4)
            child = child.next

# Function to process text
def process_text(model, text, mode='character',debug=False):
    context = Context(model.max_order, mode,debug)
    tokens = text.split() if mode == 'word' else list(text)
    for token in tokens:
        if token not in model.vocab.symbols:
            model.vocab.add_symbol(token)  # Ensure token is added to vocabulary
        symbol_id = model.vocab.get_symbol_id_or_oov(token)
        print(f"Processing token: '{token}' with ID: {symbol_id}")
        model.add_symbol_and_update(context, symbol_id)
    print("Final context:", context)
    model.print_trie()

####### 

import os
import pickle
from collections import defaultdict

# Define your Vocabulary, Context, Node, and PPMLanguageModel classes above this block

def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def train_model(text, max_order, mode, model_type='ppm',debug=False):
    vocab = Vocabulary(mode=mode,debug=debug)
    model = PPMLanguageModel(vocab, max_order,debug=debug)
    context = Context(max_order, mode,debug=debug)
    
    tokens = text.split() if mode == 'word' else list(text)
    for token in tokens:
        symbol_id = vocab.add_symbol(token)
        model.add_symbol_and_update(context, symbol_id)
    
    return model, vocab

def get_or_train_model(train_text, max_order, mode, model_type='ppm',debug=False):
    model_filename = f'{mode}_{model_type}_model.pkl'
    vocab_filename = f'{mode}_{model_type}_vocab.pkl'

    if os.path.exists(model_filename) and os.path.exists(vocab_filename):
        print(f"Loading {mode} {model_type} model from disk...")
        model = load_model(model_filename)
        vocab = load_model(vocab_filename)
    else:
        print(f"Training {mode} {model_type} model...")
        model, vocab = train_model(train_text, max_order, mode, model_type,debug)
        save_model(model, model_filename)
        save_model(vocab, vocab_filename)

    return model, vocab

def predict_next(model, vocab, initial_text, mode='character', num_predictions=5):
    if mode == 'word':
        tokens = nltk.word_tokenize(initial_text.lower())
    else:
        tokens = list(initial_text)

    context = Context(model.max_order, mode)  # Create a new context
    for token in tokens:
        symbol_id = vocab.get_symbol_id_or_oov(token)
        model.add_symbol_to_context(context, symbol_id)  # This line assumes there's a method to add symbol to context

    # Now, use the full context, not just the head
    full_context = context.get_context()
    predictions_indices = model.predict_next_indices(full_context, num_predictions)
    predictions = [vocab.get_symbol_by_id(idx) for idx in predictions_indices]

    return predictions


if __name__ == "__main__":
    vocabulary = Vocabulary(mode='word',debug=False)
    model = PPMLanguageModel(vocabulary, 5,debug=False)
    text = "hello world"
    process_text(model, text, mode='word',debug=False)
    # Now we know basics are working kets go for it   
    train_text = load_text("training_dasher.txt")
    max_order = 5  # You can adjust this as needed

    # Train or load character model
    lm_char, vocab_char = get_or_train_model(train_text, max_order, 'character',debug=False)
    initial_text = "My name is"
    print("Top 5 character predictions:", predict_next(lm_char, vocab_char, initial_text[-1], 'character'))

    # Train or load word model
    lm_word, vocab_word = get_or_train_model(train_text, max_order, 'word',debug=False)
    print("Top 5 word predictions:", predict_next(lm_word, vocab_word, initial_text, 'word'))
