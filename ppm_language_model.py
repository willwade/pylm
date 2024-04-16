
class Context:
    def __init__(self, head, order):
        self.head = head
        self.order = order


class Node:
    def __init__(self):
        self.child = None  # Leftmost child node for the current node
        self.next = None   # Next node
        self.backoff = None  # Node in the backoff structure
        self.count = 1     # Frequency count for this node
        self.symbol = None  # Symbol that this node stores

    def find_child_with_symbol(self, symbol):
        current = self.child
        while current is not None:
            if current.symbol == symbol:
                return current
            current = current.next
        return None

    def total_children_counts(self, exclusion_mask=None):
        child_node = self.child
        count = 0
        while child_node is not None:
            if exclusion_mask is None or not exclusion_mask[child_node.symbol]:
                count += child_node.count
            child_node = child_node.next
        return count

class PPMLanguageModel:

    def __init__(self, vocabulary, max_order, kn_alpha=0.49, kn_beta=0.77):
            self.vocab = vocabulary
            self.max_order = max_order
            self.knAlpha = kn_alpha
            self.knBeta = kn_beta
            self.root = Node()
            self.root.symbol = 0  # root symbol, usually vocabularies have a special root symbol
            self.num_nodes = 1
            self.use_exclusion = False

    def add_symbol_to_node(self, node, symbol):
        symbol_node = node.find_child_with_symbol(symbol)
        if symbol_node:
            # Update the counts for the given node.
            symbol_node.count += 1
        else:
            # Create a new child node and update the backoff structure.
            symbol_node = Node()
            symbol_node.symbol = symbol
            symbol_node.next = node.child
            node.child = symbol_node
            self.num_nodes += 1
            if node == self.root:
                symbol_node.backoff = self.root
            else:
                assert node.backoff is not None, "Expected valid backoff node"
                symbol_node.backoff = self.add_symbol_to_node(node.backoff, symbol)
        return symbol_node

    def create_context(self):
        return Context(self.root, 0)

    def add_symbol_to_context(self, context, symbol):
        # Check if the symbol index is within the valid range
        if symbol < 0 or symbol >= len(self.vocab.symbols):
            return  # Skip this symbol if it's out of bounds

        while context.head is not None:
            if context.order < self.max_order:
                # Extend the current context
                child_node = context.head.find_child_with_symbol(symbol)
                if child_node:
                    context.head = child_node
                    context.order += 1
                    return
            # Back off to a shorter context
            context.order -= 1
            context.head = context.head.backoff

        if context.head is None:
            context.head = self.root
            context.order = 0

    def add_symbol_and_update(self, context, symbol):
            # Ensure the symbol index is within the range of defined symbols in the vocabulary
            if symbol < 0 or symbol >= len(self.vocab.symbols):
                return  # Skip this symbol if it's out of bounds
    
            symbol_node = self.add_symbol_to_node(context.head, symbol)
            context.head = symbol_node
            context.order += 1
            while context.order > self.max_order:
                context.head = context.head.backoff
                context.order -= 1
    
    def get_probs(self, context):
        num_symbols = self.vocab.size()
        probs = [0.0] * num_symbols
        exclusion_mask = [False] * num_symbols if self.use_exclusion else None

        total_mass = 1.0
        node = context.head
        gamma = total_mass

        while node:
            count = node.total_children_counts(exclusion_mask)
            if count > 0:
                child_node = node.child
                while child_node:
                    symbol = child_node.symbol
                    if exclusion_mask is None or not exclusion_mask[symbol]:
                        p = gamma * max(child_node.count - self.knBeta, 0) / (count + self.knAlpha)
                        probs[symbol] += p
                        total_mass -= p
                        if exclusion_mask:
                            exclusion_mask[symbol] = True
                    child_node = child_node.next

            # Backoff to lower-order context
            node = node.backoff
            gamma = total_mass  # update gamma to the remaining total mass

        # Distribute remaining probability mass to unseen symbols
        num_unseen_symbols = sum(1 for seen in exclusion_mask if not seen) if exclusion_mask else num_symbols - 1
        remaining_mass = total_mass

        for i in range(1, num_symbols):  # starting from 1 to skip root symbol
            if exclusion_mask is None or not exclusion_mask[i]:
                p = remaining_mass / num_unseen_symbols
                probs[i] += p
                remaining_mass -= p

        #assert abs(sum(probs) - 1) < 1e-10, "Probabilities should sum to 1"
        print(f"Vocabulary symbols: {self.vocab.symbols}")  # Corrected reference
        print(f"Number of probabilities: {len(probs)}")
        assert len(probs) == self.vocab.size(), "Mismatch in expected probabilities length"
        return probs
    
    def print_to_console(self, node=None, indent=""):
        if node is None:
            node = self.root
            print("Root:")
        if node.child is not None:
            child = node.child
            while child:
                print(f"{indent}{child.symbol} (count: {child.count})")
                self.print_to_console(child, indent + "  ")
                child = child.next
