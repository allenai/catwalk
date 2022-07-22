from typing import List, Optional, Sequence, Tuple, Dict
from tango.common import Tqdm


class PrefixTrie():
    def __init__(self, sequences: Sequence[Sequence[int]], track_after_depth: int = 10):
        """
        Returns a PrefixTrie for ordering examples by common prefixes

        # Parameters

        sequences : `Sequence[Sequence[int]]`
            Sequences of tokens to add to the add to the Trie
        track_after_depth : `int`
            Only record sequence indices in nodes at or below this depth. This allows distinct
            sequences that coincidentally start with the first few tokens as another sequence
            not to be dropped from this barely overlapping prefix. Sequences shorter than the
            minimum depth will only have their index recorded in their final node.
        """
        self.root = PrefixTrieNode()
        self.track_after_depth = track_after_depth
        self.nodes: List['PrefixTrieNode'] = []
        for i, sequence in Tqdm.tqdm(enumerate(sequences), desc="Building PrefixTrie for caching", total=len(sequences)):
            self.add_sequence(sequence=sequence, index=i)
        # remove all indices and lenghts_covered at non-forking, non-leaf nodes
        for node in self.nodes:
            if len(node.children) == 1:
                node.indices = []
                node.lengths_covered = []
    
    def add_sequence(self, sequence: Sequence[int], index: int):
        seq_len = len(sequence)
        current_node = self.root
        for token_idx, token in enumerate(sequence):
            if token not in current_node.children:
                current_node.children[token] = PrefixTrieNode(parent=current_node, token=token)
                self.nodes.append(current_node.children[token])
            current_node = current_node.children[token]
            if (token_idx + 1 >= self.track_after_depth) or (token_idx + 1 >= seq_len):
                current_node.indices.append(index)
                current_node.lengths_covered.append(token_idx + 1)
    
    def get_leaf_nodes(self) -> List['PrefixTrieNode']:
        return [node for node in self.nodes if len(node.children) == 0]

class PrefixTrieNode():
    def __init__(self, parent: 'PrefixTrieNode' = None, token: int = None):
        self.parent = parent
        self.token = token
        self.indices: List[int] = []
        self.lengths_covered: List[int] = []
        self.children: Dict[int,'PrefixTrieNode'] = {}
    
    def get_sequence(self) -> List[Optional[int]]:
        current_node = self
        sequence = []
        while current_node.parent is not None:
            sequence.append(current_node.token)
            current_node = current_node.parent
        return sequence[::-1]
    
    def get_prefix_indices(self) -> Tuple[List[int], int]:
        """Returns all indices for subsequences of the current node including itself starting with longest and decreasing"""
        current_node = self
        indices = []
        already_found = set()
        total_lengths_covered = 0
        while current_node.parent is not None:
            new_indices = []
            for index, length_covered in zip(current_node.indices, current_node.lengths_covered):
                if index not in already_found:
                    new_indices.append(index)
                    total_lengths_covered += length_covered
            already_found.update(new_indices)
            indices.extend(new_indices)
            current_node = current_node.parent
        return indices, total_lengths_covered

if __name__ == "__main__":
    sequences = [[1,2,3],[2,3,4],[1,2,3,4]]
    trie = PrefixTrie(sequences, track_after_depth=1)
    leaves = trie.get_leaf_nodes()
    assert leaves[0].get_sequence() == [2,3,4]
    assert leaves[1].get_sequence() == [1,2,3,4]
    assert leaves[0].get_prefix_indices() == ([1], 3)
    assert leaves[1].get_prefix_indices() == ([2,0], 7)