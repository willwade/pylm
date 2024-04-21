class Vocabulary:
    def __init__(self):
        self.items_to_index = {'<OOV>': 0}  # Out-of-vocabulary has index 0
        self.index_to_items = ['<OOV>']
        self.oov_index = 0
        self.root_index = 1  # Define a root index

    def add_item(self, item):
        if item not in self.items_to_index:
            self.index_to_items.append(item)
            self.items_to_index[item] = len(self.index_to_items) - 1
        return self.items_to_index[item]

    def get_id_or_oov(self, item):
        return self.items_to_index.get(item, self.oov_index)

    def get_item_by_id(self, index):
        if 0 <= index < len(self.index_to_items):
            return self.index_to_items[index]
        return '<OOV>'

    def size(self):
        return len(self.index_to_items)

    def is_valid_id(self, index):
        return 0 <= index < self.size()

    def __len__(self):
        return self.size()

    def get_root_id(self):
        return self.root_index  # Retrieve the index for the root symbol