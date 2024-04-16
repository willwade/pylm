# python language_model_driver.py 5 path_to_train.txt path_to_test.txt

import sys
from vocabulary import Vocabulary
from ppm_language_model import PPMLanguageModel, Context
import math

def train_model(train_file, max_order):
    with open(train_file, 'r', encoding='utf-8') as file:
        contents = file.read()

    vocab = Vocabulary()
    for char in contents:
        vocab.add_symbol(char)

    lm = PPMLanguageModel(vocab, max_order)
    context = Context(lm.root, 0)
    for symbol in contents:
        lm.add_symbol_and_update(context, vocab.symbols.index(symbol))

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
                symbol = vocab.symbols.index(char) if char in vocab.symbols else -1  # OOV handling
                probs = lm.get_probs(context)
                prob = probs[symbol]
                assert prob > 0, "Invalid symbol probability"
                total_log_prob += math.log10(prob)
                num_symbols += 1
                lm.add_symbol_to_context(context, symbol)

    entropy = -(total_log_prob / num_symbols) / math.log10(2)  # converting log base 10 to base 2
    perplexity = 10 ** (-total_log_prob / num_symbols)
    print(f"Results: numSymbols = {num_symbols}, ppl = {perplexity}, entropy = {entropy} bits/char")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py max_order train_file test_file")
        sys.exit(1)

    max_order, train_file, test_file = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    lm, vocab = train_model(train_file, max_order)
    test_model(lm, vocab, test_file)
