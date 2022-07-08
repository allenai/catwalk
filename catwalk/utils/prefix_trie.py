from typing import List, Sequence, Set
from typing import Dict


class PrefixTrie():
    def __init__(self, sequences: Sequence[Sequence[int]]):
        self.root = PrefixTrieNode(parent=None, token=None)
        self.nodes = []
        for i, sequence in enumerate(sequences):
            self.add_sequence(sequence=sequence, index=i)
    
    def add_sequence(self, sequence: Sequence[int], index: int):
        current_node = self.root
        for token in sequence:
            if token not in current_node.children:
                current_node.children[token] = PrefixTrieNode(parent=current_node, token=token)
                self.nodes.append(current_node.children[token])
            current_node = current_node.children[token]
        current_node.indices.append(index)
    
    def get_leaf_nodes(self) -> List['PrefixTrieNode']:
        return [node for node in self.nodes if len(node.children) == 0]

class PrefixTrieNode():
    def __init__(self, parent: 'PrefixTrieNode', token: int):
        self.parent = parent
        self.token = token
        self.indices: List[int] = []
        self.children: Dict[int,'PrefixTrieNode'] = {}
    
    def get_sequence(self) -> List[int]:
        if self.parent is not None:
            sequence = self.parent.get_sequence()
            sequence.append(self.token)
            return sequence
        else:
            return []
    
    def get_prefix_indices(self) -> Set[int]:
        """Returns all indices for subsequences of the current node including itself"""
        if self.parent is not None:
            sequence = self.parent.get_prefix_indices()
            sequence.update(self.indices)
            return sequence
        else:
            return set()