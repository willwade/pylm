
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

    def add_child(self, symbol):
        if symbol not in self.children:
            self.children[symbol] = Node()
        return self.children[symbol]

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
    
    def iterate_children(self):
        """Yield each child node starting from the first child."""
        current = self.child
        while current:
            yield current
            current = current.next


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
        print(f"Adding symbol: {symbol} to node with symbol: {node.symbol}")
        symbol_node = node.find_child_with_symbol(symbol)
        if not symbol_node:
            symbol_node = Node()
            symbol_node.symbol = symbol
            symbol_node.next = node.child
            node.child = symbol_node
            self.num_nodes += 1
            # Set the backoff link of the new node
            symbol_node.backoff = self.find_appropriate_backoff(node, symbol)
            print(f"New node created for symbol: {symbol}")
        else:
            print(f"Node already exists for symbol: {symbol}, just updating count")

        symbol_node.count += 1
        return symbol_node
        
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
        return Context(self.root, 0)

    def add_symbol_to_context(self, context, symbol):
        print(f"Adding symbol {symbol} to context with initial head {context.head.symbol} and order {context.order}")    
    
        current = context.head
        path_found = False
    
        while not path_found and current:
            for child in current.iterate_children():
                if child.symbol == symbol:
                    context.head = child
                    context.order += 1
                    path_found = True
                    break
            if not path_found:
                current = current.backoff
    
        if not path_found:
            context.head = self.root
            context.order = 0
        print(f"Context updated: head={context.head.symbol if context.head else 'None'}, order={context.order}")
        self.print_context(context)  
        
    def add_symbol_and_update(self, context, symbol):
        print(f"Updating context and trie with symbol {symbol}")
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
        print(f"Context now at head {context.head.symbol} with order {context.order}")
        self.print_context(context) 
        return self.get_probs(context)

    def get_probs(self, context):
        print("Computing probabilities for context:")
        self.print_context(context)
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
        print(f"Intermediate probabilities: {probs}")
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
        print(f"Final probabilities: {probs}")
        assert abs(sum(probs) - 1) < 1e-10, "Probabilities should sum to 1"
        return probs

    def print_trie(self, node=None, indent=""):
        """Recursively print the trie structure from the given node."""
        if node is None:
            node = self.root
            print("Root Node:")
        
        if node.child:
            child = node.child
            while child:
                backoff_symbol = child.backoff.symbol if child.backoff else 'None'
                print(f"{indent}{child.symbol} (count: {child.count}, backoff: {backoff_symbol})")
                self.print_trie(child, indent + "  ")
                child = child.next
        else:
            print(f"{indent}Leaf node: {node.symbol} (count: {node.count})")

    def print_context(self, context):
        node = context.head
        path = []
        while node and node != self.root:
            path.append((node.symbol, node.count))
            node = node.backoff
        path.reverse()
        print("Current context path:", ' -> '.join([f"{sym}({cnt})" for sym, cnt in path]))

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
