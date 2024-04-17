class Vocabulary:
    def __init__(self):
        self.symbols = []
        self.symbol_to_index = {}
        self.root_symbol = 0
        self.root_symbol_name = "<R>"  # Name of the root symbol
        self.oov_symbol = "<OOV>"
        self.add_symbol(self.root_symbol)  # Add root symbol on initialization
        self.oov_index = self.add_symbol(self.oov_symbol)  # Add OOV symbol on initialization

    def add_symbol(self, symbol):
        if symbol not in self.symbol_to_index:
            self.symbols.append(symbol)
            index = len(self.symbols) - 1
            self.symbol_to_index[symbol] = index
            return index
        return self.symbol_to_index[symbol]

    def get_symbol_id_or_oov(self, symbol):
        return self.symbol_to_index.get(symbol, self.oov_index)

    def get_symbol_by_id(self, index):
        if 0 <= index < len(self.symbols):
            return self.symbols[index]
        return None  # or your method for handling out-of-vocabulary symbols
    
    def size(self):
        return len(self.symbols)

    def id_to_char(self, id):
        if 0 <= id < len(self.symbols):
            return self.symbols[id]
        return "<OOV>"