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
    print("Vocabulary symbols:", v.symbols)
    print("Number of probabilities:", len(probs))
    print("Initial probabilities:", probs)
    assert len(probs) == len(v.symbols), "Expected probabilities for each symbol in the vocabulary"
    
    # Enter 'a' and check the probability estimates
    lm.add_symbol_to_context(c, a_id)
    probs = lm.get_probs(c)
    print("Updated probabilities after adding 'a':", probs)
    assert probs[a_id] > 0 and probs[b_id] > 0, "Probabilities for both symbols should be greater than zero"
    assert probs[b_id] > probs[a_id], "Probability for 'b' should be more likely after adding 'a'"

    # Enter 'b'. The context becomes 'ab'. Any symbol is likely again
    lm.add_symbol_to_context(c, b_id)
    probs = lm.get_probs(c)
    print("Updated probabilities after adding 'b':", probs)
    assert probs[a_id] > 0 and probs[b_id] > 0, "Probabilities for both symbols should be greater than zero"
    assert probs[a_id] == probs[b_id], "Probabilities for both symbols should be equal after adding 'b'"

    # Re-create model to check adaptive behavior
    lm = PPMLanguageModel(v, max_order, 0.49, 0.77)
    c = lm.create_context()
    lm.add_symbol_and_update(c, a_id)
    probs = lm.get_probs(c)
    print("Probabilities after re-adding 'a':", probs)
    assert probs[a_id] > probs[b_id], "Probability for 'a' should be more likely after re-adding 'a'"

    lm.add_symbol_and_update(c, b_id)
    probs = lm.get_probs(c)
    print("Probabilities after re-adding 'b':", probs)
    assert probs[a_id] == probs[b_id], "Probabilities for both symbols should be the same after adding 'b' again"

    # Final print of the model's trie
    print("Final count trie:")
    lm.print_to_console()
    
if __name__ == '__main__':
    main()
