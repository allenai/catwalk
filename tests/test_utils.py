from catwalk.utils import PrefixTrie

def test_prefix_trie():
    sequences = [[1,2,3],[2,3,4],[1,2,3,4]]
    trie = PrefixTrie(sequences, track_after_depth=1)
    leaves = trie.get_leaf_nodes()
    assert leaves[0].get_sequence() == [2,3,4]
    assert leaves[1].get_sequence() == [1,2,3,4]
    assert leaves[0].get_subsequences() == ([1], 3)
    assert leaves[1].get_subsequences() == ([2,0], 7)