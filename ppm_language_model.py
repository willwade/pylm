
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
        count = 0
        node = self.child
        while node:
            if exclusion_mask is None or (node.symbol < len(exclusion_mask) and not exclusion_mask[node.symbol]):
                count += node.count
            node = node.next
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
        if not node:
            return None  # Safeguard against None
    
        symbol_node = node.find_child_with_symbol(symbol)
        if symbol_node:
            symbol_node.count += 1
        else:
            symbol_node = Node()
            symbol_node.symbol = symbol
            symbol_node.next = node.child
            node.child = symbol_node
            self.num_nodes += 1
    
            # Ensure backoff is properly linked
            if node == self.root:
                symbol_node.backoff = self.root
            else:
                # Safe call to potentially None backoff
                if node.backoff:
                    symbol_node.backoff = self.add_symbol_to_node(node.backoff, symbol)
                else:
                    symbol_node.backoff = None  # Or set a sensible default
    
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
        if symbol < 0 or symbol >= len(self.vocab.symbols):
            return  # Skip invalid symbols
    
        current_node = context.head
        path_found = False
    
        while context.order < self.max_order and not path_found:
            child_node = current_node.find_child_with_symbol(symbol)
            if child_node:
                child_node.count += 1  # Update existing node
                current_node = child_node
                path_found = True
            else:
                # Extend the current context if not found
                new_node = Node()
                new_node.symbol = symbol
                new_node.next = current_node.child
                current_node.child = new_node
                self.num_nodes += 1
                new_node.backoff = (current_node.backoff.find_child_with_symbol(symbol)
                                    if current_node.backoff else self.root)
                current_node = new_node
    
            context.head = current_node
            context.order += 1
    
        return self.get_probs(context)

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
                        adjusted_count = max(child_node.count - self.knBeta, 0)
                        p = gamma * adjusted_count / (count + self.knAlpha)
                        probs[symbol] += p
                        total_mass -= p
                        if exclusion_mask:
                            exclusion_mask[symbol] = True
                    child_node = child_node.next
    
            node = node.backoff
            gamma = total_mass
    
        # Distribute remaining probability mass to unseen symbols
        unseen_symbols = [i for i in range(1, num_symbols) if exclusion_mask is None or not exclusion_mask[i]]
        if unseen_symbols:
            for i in unseen_symbols:
                p = total_mass / len(unseen_symbols)
                probs[i] += p
                total_mass -= p
        
        # Verify that total_mass is non-zero before distribution to avoid division by zero errors
        # Also, handle the case where there are no unseen symbols to distribute the remaining mass
        if total_mass > 0:
            print("Final total probability mass after distribution:", total_mass)  # Debug output
            print("Individual Probabilities:", probs)  # Debug output
        else:
            print("No remaining probability mass to distribute.")        
        
        print("Final total probability mass after distribution:", total_mass)  # Debug output
        print("Individual Probabilities:", probs)  # Debug output
    
        assert abs(sum(probs) - 1) < 1e-10, "Probabilities should sum to 1"
        return probs


    def print_to_console(self, node=None, indent=""):
        if node is None:
            node = self.root
            print("Root:")
        if node.child is not None:
            child = node.child
            while child:
                backoff_id = child.backoff.symbol if child.backoff else "None"
                print(f"{indent}{child.symbol} (count: {child.count}, backoff: {backoff_id})")
                self.print_to_console(child, indent + "  ")
                child = child.next
