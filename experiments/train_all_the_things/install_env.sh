#!/bin/bash

set -xeo pipefail

if [[ -d /net/nfs2.allennlp/dirkg/cache ]]; then
  rm -rf ~/.cache
  ln -s /net/nfs2.allennlp/dirkg/cache ~/.cache
elif [[ -d /net/nfs.cirrascale/allennlp/dirkg/cache ]]; then
  rm -rf ~/.cache
  ln -s /net/nfs.cirrascale/allennlp/dirkg/cache ~/.cache
fi

pip install -r requirements.txt