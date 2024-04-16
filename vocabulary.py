class Vocabulary:
    def __init__(self):
        self.symbols = [] 

    def add_symbol(self, symbol):
        if symbol in self.symbols:
            return self.symbols.index(symbol)
        self.symbols.append(symbol)
        return self.symbols.index(symbol)

    def size(self):
        return len(self.symbols)

