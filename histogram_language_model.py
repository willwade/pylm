import math
from typing import Optional, List

class Context:
    """ Since the histogram models are context-less, this class is an empty handle to comply with the interface of context-based models. """
    pass

class HistogramLanguageModel:
    def __init__(self, vocab):
        assert vocab.size() > 1, "Expecting at least two symbols in the vocabulary"
        self.vocab = vocab
        self.total_observations = 0
        self.histogram = [0] * self.vocab.size()

    def create_context(self) -> Optional[Context]:
        """ Creates new context which is initially empty. """
        return Context()

    def clone_context(self, context: Optional[Context]) -> Optional[Context]:
        """ Clones existing context. """
        return Context()

    def add_symbol_to_context(self, context: Optional[Context], symbol: int):
        """ Adds symbol to the supplied context. Does not update the model. """
        pass

    def add_symbol_and_update(self, context: Optional[Context], symbol: int):
        if symbol <= self.vocab.get_root_id():  # Only add valid symbols.
            return
        assert symbol < self.vocab.size(), f"Invalid symbol: {symbol}"
        self.histogram[symbol] += 1
        self.total_observations += 1

    def get_probs(self, context: Optional[Context]) -> Optional[List[float]]:
        num_symbols = self.vocab.size()
        num_valid_symbols = num_symbols - 1  # Minus the first symbol.
        probs = [0.0] * num_symbols  # Ignore first symbol.

        # Figure out the number of unique (seen) symbols.
        num_unique_seen_symbols = sum(1 for x in self.histogram if x > 0)

        # Compute the distribution.
        py_alpha = 0.50
        py_beta = 0.77
        epsilon = 1E-12
        denominator = self.total_observations + py_alpha
        base_factor = (py_alpha + py_beta * num_unique_seen_symbols) / denominator
        uniform_prior = 1.0 / num_valid_symbols
        total_mass = 1.0
        for i in range(1, num_symbols):
            empirical = (self.histogram[i] - py_beta) / denominator if self.histogram[i] > 0 else 0.0
            probs[i] = empirical + base_factor * uniform_prior
            total_mass -= probs[i]

        assert math.isclose(total_mass, 0, abs_tol=epsilon), f"Invalid remaining probability mass: {total_mass}"

        # Adjust the remaining probability mass, if any.
        left_symbols = num_valid_symbols
        for i in range(1, num_symbols):
            p = total_mass / left_symbols
            probs[i] += p
            total_mass -= p
            left_symbols -= 1

        assert total_mass == 0, "Expected remaining probability mass to be zero!"
        assert math.isclose(sum(probs), 1.0, abs_tol=epsilon)
        return probs

    def print_to_console(self):
        print("Histogram of counts (total: " + str(self.total_observations) + "): ")
        for i in range(1, len(self.histogram)):  # Starts from 1 assuming 0 is the root or OOV which we might want to skip
            print("\t" + self.vocab.get_item_by_id(i) + ": " + str(self.histogram[i]))


