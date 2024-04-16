import sys
from vocabulary import Vocabulary
from ppm_language_model import PPMLanguageModel

def main():
    # Create a small vocabulary
    v = Vocabulary()
    a_id = v.add_symbol("a")
    b_id = v.add_symbol("b")

    # Build the PPM language model trie and update the counts
    max_order = 5
    lm = PPMLanguageModel(v, max_order, 0.49, 0.77)  # knAlpha and knBeta are set here
    c = lm.create_context()
    lm.add_symbol_and_update(c, a_id)
    lm.add_symbol_and_update(c, b_id)

    print("Initial count trie:")
    lm.print_to_console()
    
    # Check static (non-adaptive) mode
    c = lm.create_context()
    probs = lm.get_probs(c)
    assert len(probs) == len(v.symbols), "Expected probabilities for each symbol in the vocabulary"
    print(probs)

    # Enter 'a' and check the probability estimates
    lm.add_symbol_to_context(c, a_id)
    probs = lm.get_probs(c)
    print(probs)
    assert probs[a_id] > 0 and probs[b_id] > 0, "Probabilities for both symbols should be greater than zero"
    
    # Enter 'b' and check probabilities again.  The context becomes 'ab'. Any symbol is likely again
    lm.add_symbol_to_context(c, b_id)
    probs = lm.get_probs(c)
    print(probs)
    assert probs[a_id] > 0 and probs[b_id] > 0, "Probabilities for both symbols should be greater than zero after 'b'"

    # Check adaptive mode in which the model is updated as symbols are entered
    lm = PPMLanguageModel(v, max_order, 0.49, 0.77)  # Re-create
    c = lm.create_context()
    lm.add_symbol_and_update(c, a_id)
    probs = lm.get_probs(c)
    assert probs[a_id] > 0 and probs[a_id] > probs[b_id], "Probability for 'a' should be more likely"
    print(probs)

    # Enter 'b' and update the model
    lm.add_symbol_and_update(c, b_id)
    probs = lm.get_probs(c)
    assert probs[a_id] > 0 and probs[a_id] == probs[b_id], "Probabilities for both symbols should be the same"
    print(probs)

if __name__ == '__main__':
    main()
