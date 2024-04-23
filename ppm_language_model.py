import math
"""
   @fileoverview Prediction by Partial Matching (PPM) language model.
  
   The original PPM algorithm is described in [1]. This particular
   implementation has been inspired by the PPM model used by Dasher, an
   Augmentative and alternative communication (AAC) input method developed by
   the Inference Group at University of Cambridge. The overview of the system
   is provided in [2]. The details of this algorithm, which is different from
   the standard PPM, are outlined in general terms in [3]. Please also see [4]
   for an excellent overview of various PPM variants.
  
   References:
   -----------
     [1] Cleary, John G. and Witten, Ian H. (1984): Data Compression Using
         Adaptive Coding and Partial String Matching, IEEE Transactions on
         Communications, vol. 32, no. 4, pp. 396402.
     [2] Ward, David J. and Blackwell, Alan F. and MacKay, David J. C. (2000):
         Dasher - A Data Entry Interface Using Continuous Gestures and
         Language Models, UIST'00 Proceedings of the 13th annual ACM symposium
         on User interface software and technology, pp. 129137, November, San
         Diego, USA.
     [3] Cowans, Phil (2005): Language Modelling In Dasher -- A Tutorial,
         June, Inference Lab, Cambridge University (presentation).
     [4] Jin Hu Huang and David Powers (2004): "Adaptive Compression-based
         Approach for Chinese Pinyin Input." Proceedings of the Third SIGHAN
         Workshop on Chinese Language Processing, pp. 24--27, Barcelona, Spain,
         ACL.
   Please also consult the references in README.md file in this directory.
"""


class Context:
    """
    Handle encapsulating the search context.
    """

    def __init__(self, head, order, debug=False):
        """
        Constructor.
        @param {?Node} head Head node of the context.
        @param {number} order Length of the context.
        """
        self.head = head
        self.order = order
        self.debug = debug

    def get_current_head(self):
        """
        Returns the node that currently represents the head of the context.
        Useful for debugging or for operations that need to know the current state of the context.
        """
        return self.head

    def reset_to_root(self):
        self.head = self.root
        self.order = 0


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
        self.children = {}  # Use a dictionary to hold child nodes
        self.backoff = None  # Node in the backoff structure
        self.count = 1       # Frequency count for this node
        self.symbol = symbol

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
        return self.children.get(symbol, None)

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
        return sum(node.count for sym, node in self.children.items()
                   if exclusion_mask is None or (sym < len(exclusion_mask) and not exclusion_mask[sym]))

    def num_distinct_symbols(self):
        """
        Returns the number of distinct symbols (children) for this node.
        """
        return len(self.children)


class PPMLanguageModel:
    def __init__(self, vocabulary, max_order, kn_alpha=0.49, kn_beta=0.77, debug=False):
        """
        Initializes a PPM Language Model.

        Args:
            vocab (Vocabulary): Symbol vocabulary object.
            max_order (int): Maximum length of the context.
            kn_alpha (int):Kneser-Ney "-like" smoothing parameters.
            kn_beta(int): Kneser-Ney "-like" smoothing parameters.
        """
        assert vocabulary.size() > 1, "Vocabulary must contain at least two symbols"
        if max_order < 1:
            raise ValueError("max_order must be a positive integer")
        self.vocab = vocabulary
        self.max_order = max_order
        self.kn_alpha = kn_alpha
        self.kn_beta = kn_beta
        self.epsilon = 1e-10
        self.root = Node()
        self.root.symbol = 0  # root symbol, usually vocabularies have a special root symbol
        self.num_nodes = 1
        self.use_exclusion = False
        self.debug = debug

    def add_symbol_to_node(self, node, symbol):
        """
        Adds symbol to the supplied node.
        @param {?Node} node Tree node which to grow.
        @param {number} symbol Symbol.
        @return {?Node} Node with the symbol.
        """
        if symbol not in node.children:
            new_node = Node()
            new_node.backoff = self.find_appropriate_backoff(node, symbol)
            node.children[symbol] = new_node
            self.num_nodes += 1  # Increment the number of nodes
            if self.debug:
                print(
                    f"Debug: New node created for symbol {symbol} with backoff pointing to {new_node.backoff}, Total nodes: {self.num_nodes}")
        else:
            node.children[symbol].count += 1
            if self.debug:
                print(
                    f"Debug: Revisiting existing node for symbol {symbol} with new count {node.children[symbol].count}")

        return node.children[symbol]

    def find_appropriate_backoff(self, node, symbol):
        current = node.backoff
        while current is not None:
            if symbol in current.children:
                return current.children[symbol]
            current = current.backoff
        return self.root  # Ensure this is really the desired fallback

    def context_to_node(self, context):
        node = self.root  # Starting at the root of the trie
        for char in context:
            if char in node.children:
                node = node.children[char]
            else:
                return None  # Context does not exist in the trie
        return node

    def create_context(self):
        """
        Creates new context which is initially empty.
        @return {?Context} Context object.
        """
        return Context(self.root, 0)

    def clone_context(context):
        """
        Clones existing context.
        @param {?Context} context Existing context object.
        @return {?Context} Cloned context object.
        """
        return Context(head=context.head, order=context.order)

    def add_symbol_to_context(self, context, symbol):
        if not (0 < symbol < self.vocab.size()):
            if self.debug:
                print(f"Invalid symbol: {symbol}")
            return  # Only add valid symbols

        if self.debug:
            print(
                f"Starting add_symbol_to_context with symbol ID {symbol} ({self.vocab.get_item_by_id(symbol)})")

        while context.head is not None:
            if context.order < self.max_order:
                child_node = context.head.find_child_with_symbol(symbol)
                if child_node is not None:
                    context.head = child_node
                    context.order += 1
                    if self.debug:
                        print(
                            f"Extended context to node with symbol '{self.vocab.get_item_by_id(symbol)}', new order {context.order}")
                    return  # Successfully extended the context

            # Back off to shorter context
            if self.debug:
                print(
                    f"No child node found for symbol '{self.vocab.get_item_by_id(symbol)}', backoff to shorter context")
            context.order -= 1
            context.head = context.head.backoff

        # If no valid head is found, reset context
        if context.head is None:
            context.head = self.root
            context.order = 0
            if self.debug:
                print("Context reset to root due to null head after backoff")

    def add_symbol_and_update(self, context, symbol):
        if self.debug:
            print(
                f"Attempting to add/update symbol: {symbol} ({self.vocab.get_item_by_id(symbol)})")

        if not self.vocab.is_valid_id(symbol):
            if self.debug:
                print("Invalid symbol ID.")
            return

        current_node = context.head
        found = False

        # Traverse the trie to find or create the needed node
        while context.head is not None and context.order < self.max_order:
            child_node = self.add_symbol_to_node(context.head, symbol)

            # Update the context head to point to the child node
            context.head = child_node
            context.order += 1
            if self.debug:
                print(
                    f"Context updated: head ID = {id(context.head)}, order = {context.order}")
            return

        if not found:
            # Reset to root if no suitable child is found and we've reached max order
            context.head = self.root
            context.order = 0

        if self.debug:
            print(
                f"Context updated: head ID = {id(context.head)}, order = {context.order}")

    def debug_node_details(self, node):
        # Simplified to use actual Node properties
        if node:
            return f"Node at {id(node)}, Count={node.count}, Children={len(node.children)}"
        return "None"

    def get_probs(self, context):
        num_symbols = self.vocab.size()
        probs = [0.0] * num_symbols
        exclusion_mask = [False] * num_symbols if self.use_exclusion else None
        total_mass = 1.0
        node = context.get_current_head()

        while node:
            count = node.total_children_counts(exclusion_mask)
            if count > 0:
                for symbol, child_node in node.children.items():
                    if exclusion_mask is None or not exclusion_mask[symbol]:
                        p = max(child_node.count - self.kn_beta, 0) / \
                            (count + self.kn_alpha)
                        adjusted_p = p * total_mass
                        probs[symbol] += adjusted_p
                        if exclusion_mask:
                            exclusion_mask[symbol] = True

            # Calculate remaining probability mass for backing off
            backoff_mass = self.kn_alpha / \
                (count + self.kn_alpha) if count > 0 else 1.0
            total_mass *= backoff_mass
            node = node.backoff

        # Adjust probabilities to sum to 1
        self.normalize_probs(probs, exclusion_mask)
        return probs

    def normalize_probs(self, probs, exclusion_mask):
        num_symbols = self.vocab.size()
        remaining_mass = 1.0 - sum(probs)

        if remaining_mass > 0:
            non_excluded_count = sum(1 for i in range(
                num_symbols) if exclusion_mask is None or not exclusion_mask[i])
            for i in range(num_symbols):
                if exclusion_mask is None or not exclusion_mask[i]:
                    probs[i] += remaining_mass / non_excluded_count

        # Ensure probabilities sum exactly to 1.0, adjusting for small floating-point discrepancies
        sum_probs = sum(probs)
        if not math.isclose(sum_probs, 1.0, abs_tol=self.epsilon):
            adjustment = (1.0 - sum_probs) / num_symbols
            probs = [p + adjustment for p in probs]

        assert math.isclose(sum(
            probs), 1.0, abs_tol=self.epsilon), "Probabilities do not sum to 1 after normalization"

    def calculate_gamma(self, node, count, exclusion_mask):
        if count > 0:
            num_extensions = sum(1 for sym in range(
                self.vocab.size()) if exclusion_mask is None or not exclusion_mask[sym])
            return (self.kn_alpha + num_extensions * self.kn_beta) / (count + self.kn_alpha)
        return 1.0

    def get_probs_with_symbols(lm, context):
        # Retrieve the raw probabilities from the model
        raw_probs = lm.get_probs(context)

        # Get the symbols corresponding to the indices
        symbols = [lm.vocab.get_item_by_id(i) for i in range(len(raw_probs))]

        # Pair each symbol with its probability
        prob_symbol_pairs = list(zip(symbols, raw_probs))

        return prob_symbol_pairs

    def predict_next_ids(self, context, num_predictions=1):
        if self.debug:
            print(
                f"Debug: Current context before prediction: {context}, Order: {context.order}")

        if not context.head or context.order < 1:
            print("Debug: Invalid context or order.")  # Additional debug
            return []

        probs = self.get_probs(context)
        if not probs or all(p == 0 for p in probs):
            if self.debug:
                print("Debug: No valid probabilities computed.",
                      probs)  # Additional debug
            return []
        if self.debug:
            # Print raw probabilities
            print(f"Debug: Raw probabilities: {probs}")
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[
            :num_predictions]
        top_predictions = [(index, probs[index])
                           for index in top_indices if probs[index] > 0]
        if self.debug:
            print("Debug: Predictions computed:",
                  top_predictions)  # Additional debug
        return top_predictions

    def print_trie(self, node=None, indent=""):
        if node is None:
            node = self.root

        stack = [(node, None, indent)]

        while stack:
            current_node, symbol, current_indent = stack.pop()

            if current_node == self.root:
                print(f"{current_indent}<Root> (count: {current_node.count})")
            else:
                if symbol is not None:
                    symbol_char = self.vocab.get_item_by_id(symbol)
                    if symbol_char == ' ':  # Handling space for clearer output
                        symbol_char = '<space>'
                    elif symbol_char is None:  # Handling None or unexpected values
                        symbol_char = '<undefined>'
                else:
                    symbol_char = '<undefined>'

                backoff_char = "<None>"
                if current_node.backoff:
                    backoff_symbol = current_node.backoff.symbol if hasattr(
                        current_node.backoff, 'symbol') else None
                    backoff_char = self.vocab.get_item_by_id(
                        backoff_symbol) if backoff_symbol is not None else "<None>"
                    if backoff_char == ' ':
                        backoff_char = '<space>'
                    elif backoff_char is None:
                        backoff_char = '<undefined>'

                print(
                    f"{current_indent}{symbol_char} (count: {current_node.count}, backoff: {backoff_char})")

            for sym, child in reversed(list(current_node.children.items())):
                stack.append((child, sym, current_indent + "  "))

    def print_context(self, context):
        node = context.head
        path = []
        while node and node != self.root:
            path.append((node.symbol, node.count))
            node = node.backoff
        path.reverse()
        print("Current context path:",
              ' -> '.join([f"{sym}({cnt})" for sym, cnt in path]))

    def print_to_console(self):
        self.print_trie(self.root)
