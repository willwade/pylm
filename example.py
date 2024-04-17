from vocabulary import Vocabulary
from ppm_language_model import PPMLanguageModel
from histogram_language_model import HistogramLanguageModel
from polya_tree_language_model import PolyaTreeLanguageModel

def test_ppm_language_model(vocab):
    max_order = 5
    lm = PPMLanguageModel(vocab, max_order)
    context = lm.create_context()
    a_id = vocab.get_symbol_id_or_oov("a")
    b_id = vocab.get_symbol_id_or_oov("b")

    # Update model with sequence "ab"
    lm.add_symbol_and_update(context, a_id)
    lm.add_symbol_and_update(context, b_id)
    print("Initial count trie:")
    lm.print_to_console()

    # Static mode check
    context = lm.create_context()
    probs = lm.get_probs(context)
    print("Probabilities after initialization:", probs)

    # Enter 'a' and check updates
    lm.add_symbol_to_context(context, a_id)
    probs = lm.get_probs(context)
    print("Probabilities after 'a':", probs)

    # Re-creation for adaptive mode
    lm = PPMLanguageModel(vocab, max_order)
    context = lm.create_context()
    lm.add_symbol_and_update(context, a_id)
    lm.add_symbol_and_update(context, b_id)
    lm.add_symbol_and_update(context, b_id)  # Added 'b' twice to model 'abb'
    print("Final count trie after adaptive updates:")
    lm.print_to_console()

def test_histogram_language_model(vocab):
    lm = HistogramLanguageModel(vocab)
    context = lm.create_context()
    training_data = "ababababab"
    for symbol in training_data:
        lm.add_symbol_and_update(context, vocab.get_symbol_id_or_oov(symbol))
    print("Histogram after training:")
    lm.print_to_console()

def test_polya_tree_language_model(vocab):
    lm = PolyaTreeLanguageModel(vocab)
    context = lm.create_context()
    for symbol in "aacccdd":
        lm.add_symbol_and_update(context, vocab.get_symbol_id_or_oov(symbol))
    print("Polya tree model probabilities:")
    probs = lm.get_probs(context)
    print(probs)

def main():
    vocab = Vocabulary()
    vocab.add_symbol("a")
    vocab.add_symbol("b")

    print("Testing PPM Language Model:")
    test_ppm_language_model(vocab)

    print("Testing Histogram Language Model:")
    test_histogram_language_model(vocab)

    vocab.add_symbol("c")
    vocab.add_symbol("d")
    print("Testing Polya Tree Language Model:")
    test_polya_tree_language_model(vocab)

if __name__ == "__main__":
    main()
