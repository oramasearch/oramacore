#!/bin/bash

set -axe

CACHE_FOLDER=.custom_models/sentenceTransformersParaphraseMultilingualMiniLML12v2
mkdir -p $CACHE_FOLDER

if [ ! -e $CACHE_FOLDER/model.onnx ]; then
    curl -L 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/onnx/model.onnx?download=true' \
        -o $CACHE_FOLDER/model.onnx
fi

if [ ! -e $CACHE_FOLDER/config.json ]; then
    curl -L 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/config.json?download=true' \
        -o $CACHE_FOLDER/config.json 
fi

if [ ! -e $CACHE_FOLDER/special_tokens_map.json ]; then
    curl -L 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/special_tokens_map.json?download=true' \
        -o $CACHE_FOLDER/special_tokens_map.json
fi

if [ ! -e $CACHE_FOLDER/tokenizer.json ]; then
    curl -L 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/tokenizer.json?download=true' \
        -o $CACHE_FOLDER/tokenizer.json
fi

if [ ! -e $CACHE_FOLDER/tokenizer_config.json ]; then
    curl -L 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/tokenizer_config.json?download=true' \
        -o $CACHE_FOLDER/tokenizer_config.json
fi
