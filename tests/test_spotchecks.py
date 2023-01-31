import sys

import catwalk.__main__


def test_squad():
    args = catwalk.__main__._parser.parse_args([
        "--model", "bert-base-uncased",
        "--task", "squad",
        "--split", "validation",
        "--limit", "100"
    ])
    catwalk.__main__.main(args)
