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
    def __init__(self):
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
        total = sum(node.count for sym, node in self.children.items()
                    if exclusion_mask is None or not exclusion_mask[sym])
        return total

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
        self.vocab = vocabulary
        self.max_order = max_order
        self.knAlpha = kn_alpha
        self.knBeta = kn_beta
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
            node.children[symbol] = Node()
            node.children[symbol].symbol = symbol
            self.num_nodes += 1
            # Set the backoff link of the new node
            node.children[symbol].backoff = self.find_appropriate_backoff(node, symbol)
        else:
            node.children[symbol].count += 1
    
        return node.children[symbol]

        
    def find_appropriate_backoff(self, node, symbol):
        """
        Traverse the backoff chain from the given node to find a node that has a child with the given symbol.
        If such a node is found, its child node for the symbol is returned as the backoff node.
        If no such node is found in the chain, backoff to the root.
        """
        current = node.backoff  # Start from the parent node's backoff
        while current is not None:  # Traverse back up the trie through the backoff links
            child = current.find_child_with_symbol(symbol)
            if child:
                return child  # Found a valid backoff node
            current = current.backoff
        return self.root  # Default backoff to the root if no appropriate node is found

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
        """
        Adds symbol to the supplied context. Does not update the model.
        @param {?Context} context Context object.
        @param {number} symbol Integer symbol.
        """    
        if symbol in context.head.children:
            context.head = context.head.children[symbol]
            context.order += 1
        else:
            context.head = self.root
            context.order = 0
        if self.debug:
            self.print_context(context)
        
    def add_symbol_and_update(self, context, symbol):
        """
        Adds symbol to the supplied context and updates the model.
        @param {?Context} context Context object.
        @param {number} symbol Integer symbol.
        """    
        #print(f"Updating context and trie with symbol {symbol}")
        
        # Handle invalid symbols
        if symbol < 0 or symbol >= self.vocab.size():
            return
        
        # Add or retrieve the node for this symbol
        symbol_node = self.add_symbol_to_node(context.head, symbol)
        assert symbol_node, "Failed to add or find a node for the symbol"
        
        # Update context to point to this node
        context.head = symbol_node
        context.order += 1
        #print(f"Context now at head {context.head.symbol} with order {context.order}")
        
        # Reduce context order if it exceeds max_order
        while context.order > self.max_order:
            context.head = context.head.backoff
            context.order -= 1
            #print(f"Reducing context order, now {context.order}")
    
    def get_probs(self, context):
        """
        Returns probabilities for all the symbols in the vocabulary given the context.
        """
        #print("Computing probabilities for context:")
    
        num_symbols = self.vocab.size()
        probs = [0.0] * num_symbols
        exclusion_mask = [False] * num_symbols if self.use_exclusion else None
        total_mass = 1.0
        node = context.head
        gamma = total_mass
    
        while node is not None:
            count = node.total_children_counts(exclusion_mask)
            if count > 0:
                for symbol, child_node in node.children.items():
                    if not exclusion_mask or not exclusion_mask[symbol]:
                        p = gamma * max(child_node.count - self.knBeta, 0) / (count + self.knAlpha)
                        probs[symbol] += p
                        total_mass -= p
                        if exclusion_mask:
                            exclusion_mask[symbol] = True
            gamma = self.calculate_gamma(node, count, exclusion_mask)
            node = node.backoff
        
        assert total_mass >= 0, "Invalid remaining probability mass: {}".format(total_mass)
        self.normalize_probs(probs, num_symbols, exclusion_mask, total_mass)
        return probs
    
    def calculate_gamma(self, node, count, exclusion_mask):
        q = sum(not mask for mask in exclusion_mask) if exclusion_mask else node.num_distinct_symbols()
        return (self.knBeta * q + self.knAlpha) / (count + self.knAlpha)
    
    def normalize_probs(self, probs, num_symbols, exclusion_mask, total_mass):
        if exclusion_mask is not None:
            num_unseen_symbols = sum(1 for i in range(1, num_symbols) if not exclusion_mask[i])
        else:
            num_unseen_symbols = num_symbols - 1  # Exclude the root symbol at index 0
    
        remaining_mass = total_mass
        # Apply remaining mass proportionally to non-excluded symbols
        for i in range(1, num_symbols):
            if exclusion_mask is None or not exclusion_mask[i]:  # Check if symbol is not excluded
                if num_unseen_symbols > 0:  # Avoid division by zero
                    p = remaining_mass / num_unseen_symbols
                    probs[i] += p
                    remaining_mass -= p
    
        # Direct normalization to ensure sum of probabilities is 1.0
        sum_probs = sum(probs)
        if sum_probs > 0:
            probs = [p / sum_probs for p in probs]
    
        # Assert that probabilities sum to 1 (with a small tolerance for floating-point arithmetic)
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-10), "Probs do not sum to 1"
        return probs
    

    def print_trie(self, node=None, indent=""):
        """Recursively print the trie structure from the given node with detailed formatting."""
        if node is None:
            node = self.root
            print(f"{indent}<R>({node.symbol}) [{node.count}]")  # Assuming root has a default symbol ID and count
            indent += "  "
        else:
            # Translate symbol ID to character for output, assuming a method to do so exists
            symbol_char = self.vocab.id_to_char(node.symbol) if node.symbol >= 0 else "<OOV>"
            print(f"{indent}{symbol_char}({node.symbol}) [{node.count}]")
        
        for symbol, child in node.children.items():
            self.print_trie(child, indent + "  ")
        
        if not node.children:  # This node is a leaf
            print(f"{indent}Leaf node: {symbol_char}({node.symbol}) [{node.count}]")

    def print_context(self, context):
        node = context.head
        path = []
        while node and node != self.root:
            path.append((node.symbol, node.count))
            node = node.backoff
        path.reverse()
        print("Current context path:", ' -> '.join([f"{sym}({cnt})" for sym, cnt in path]))

    def print_to_console(self):
        self.print_trie(self.root)