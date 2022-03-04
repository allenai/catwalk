from collections import Counter

def get_majority(lst):
    c = Counter(lst)
    rank = c.most_common()
    if len(rank) == 1:
        return rank[0][0]
    elif rank[0][1] == rank[1][1]:
        return None
    else:
        return rank[0][0]
