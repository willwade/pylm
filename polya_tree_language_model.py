"""
  @fileoverview Pólya Tree (PT) language model.

  The Pólya tree method uses a balanced binary search tree whose leaf nodes
  contain the symbols of the vocabulary V, such that each symbol v \in V can be
  identified by a sequence of (at most \log_2(|V|)) binary branching decisions
  from the root of the tree. The leafs of the tree represent predictive
  probabilities for the respective symbols. The tree has N=|V|−1 internal
  nodes, each containing a value \theta_i that represents the probability of
  choosing between its two children [1]. The probability of a given symbol v is
  then defined as [2]:

    P(v) = \prod_{i \in PATH(v)} Bernoulli(b_i | \theta_i)
         = \prod_{i \in PATH(v)} \theta_i^{b_i} (1 - \theta_i)^{1 - b_i} ,

  where PATH(v) denotes the set of nodes i that belong to the path from the
  root to node v, b_i \in {0, 1} is the branching decision at node i and
  \theta_i is the Bernoulli distribution bias at node i. See documentation of
  the getProbs() API to see how the \theta is approximated via conjugate priors
  using beta distribution, in other words, \theta_i ~ Beta(\alpha, \beta).

  This language model can be used as a prior in a more sophisticated
  context-based model.

  References:
  -----------
    [1] Steinruecken, Christian (2015): "Lossless Data Compression", PhD
        dissertation, University of Cambridge.
    [2] Gleave, Adam and Steinruecken, Christian (2017): "Making compression
        algorithms for Unicode text", arXiv preprint arXiv:1701.04047.
    [3] Mauldin, R. Daniel and Sudderth, William D. and Williams, S. C. (1992):
        "Polya Trees and Random Distributions", The Annals of Statistics,
        pp. 1203--1221.
"""


import math
from typing import List, Optional

class Node:
    def __init__(self):
        self.num_branch_left = 0
        self.num_branch_right = 0

class PathNode:
    def __init__(self):
        self.id = 0
        self.left_branch = False

class Context:
    pass

class PolyaTreeLanguageModel:
    def __init__(self, vocab):
        assert vocab.size() > 1, "Expecting at least two symbols in the vocabulary"
        self.vocab = vocab
        self.total_observations = 0
        self.nodes = None
        self.root_probs = [0.0] * (self.vocab.size() - 1)
        self.beta_distr_alpha = 0.5  # Class attribute for alpha
        self.beta_distr_beta = 0.5   # Class attribute for beta
        self.build_tree()

    def create_context(self) -> Optional[Context]:
        return Context()

    def clone_context(self, context: Optional[Context]) -> Optional[Context]:
        return Context()

    def add_symbol_to_context(self, context: Optional[Context], symbol: int):
        pass

    def add_symbol_and_update(self, context: Optional[Context], symbol: int):
        if symbol <= self.vocab.root_symbol:
            return
        assert symbol < self.vocab.size(), f"Invalid symbol: {symbol}"
        path = self.get_path(symbol)
        assert len(path) > 1, f"Expected more than one node in the path for symbol {symbol}"
        num_internal_nodes = len(path) - 1
        for i in range(num_internal_nodes):
            path_node = path[i]
            tree_node = self.nodes[path_node.id]
            if path_node.left_branch:
                tree_node.num_branch_left += 1
            else:
                tree_node.num_branch_right += 1
        self.total_observations += 1

    def get_probs(self, context: Optional[Context]) -> Optional[List[float]]:
        num_symbols = self.vocab.size()
        probs = [0.0] * num_symbols
        for i in range(1, num_symbols):
            path = self.get_path(i)
            # Adjust the index by -1 to account for the root symbol exclusion in root_probs
            probs[i] = self.root_probs[i - 1]  
            for path_node in path:
                tree_node = self.nodes[path_node.id]
                theta = (self.beta_distr_alpha + tree_node.num_branch_left) / \
                        (self.beta_distr_alpha + self.beta_distr_beta + tree_node.num_branch_left + tree_node.num_branch_right)
                probs[i] *= theta if path_node.left_branch else (1 - theta)
        return probs

    def build_tree(self):
            num_symbols = self.vocab.size()
            # Calculate the number of nodes in the tree
            num_nodes = 2 * num_symbols - 1  # Includes internal nodes and leaves
            self.nodes = [Node() for _ in range(num_nodes)]
            
            # Parameters for the beta distribution
            theta = self.beta_distr_alpha / (self.beta_distr_alpha + self.beta_distr_beta)
    
            # Calculate probabilities for each symbol except the root
            for i in range(1, num_symbols):  # Start from 1 to skip the root symbol
                path = self.get_path(i)
                p = 1.0
                for path_node in path:
                    if path_node.left_branch:
                        p *= theta
                    else:
                        p *= (1.0 - theta)
                # Save probability, adjusting index by -1 to account for root symbol
                self.root_probs[i - 1] = p

    def get_path(self, symbol: int) -> List[PathNode]:
        num_symbols = self.vocab.size() - 1
        symbol_node_id = num_symbols - 1 + symbol - 1
        path = []
        parent = (symbol_node_id - 1) // 2
        while parent >= 1:
            path.append(parent)
            parent = (parent - 1) // 2
        path.append(0)
        path.reverse()
        nodes = [PathNode() for _ in path]
        for i in range(len(path)):
            node = nodes[i]
            node.id = path[i]
            if i > 0:
                node.left_branch = node.id == 2 * path[i - 1] + 1
        return nodes

    def print_to_console(self):
        for node_id in range(len(self.nodes)):
            node = self.nodes[node_id]
            print(f"Node {node_id}: Left {node.num_branch_left}, Right {node.num_branch_right}")

# Assuming a `Vocabulary` class with methods like `size()` and properties like `root_symbol`.
