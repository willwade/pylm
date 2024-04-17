class Vocabulary:
    def __init__(self):
        self.symbols = []
        self.symbol_to_index = {} 

    def add_symbol(self, symbol):
        if symbol not in self.symbol_to_index:
            self.symbols.append(symbol)
            self.symbol_to_index[symbol] = len(self.symbols) - 1
        return self.symbol_to_index[symbol]

    def size(self):
        return len(self.symbols)

