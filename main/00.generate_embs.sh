#!/bin/bash

LANG=han
DATA_DOMAIN=None      # Data domain, choose from [Newsfirst, ITN, Army, Hiru]

TKPERT_J=16                # J for TK-PERT
TKPER_SHAPE=20             # Shape for TK-PERT

SPLIT_METHOD=ofls          # Split method, choose from [sbs, ofls]
# if ofls has been chosen
FL=30                      # Fixed-Length for ofls
OVERLAP=0.5                # Overlapping Rate for ofls

DATA_PATH=../data
OUT_PATH=../embs             # Path to save the embeddings
mkdir -p $OUT_PATH

CHUNK_SIZE=2048              # Chunk size used to split document segments into batches for embedding
SAVE_NUM=4096                # Document Num. for saving in one embedding file

INPUT_FORMAT=txt             # Input file extension

echo "Begin generating embeddings..."
python ../utils/generate_embs.py
    --seg_len "$FL" \
    --overlap "$OVERLAP" \
    --split_method "$SPLIT_METHOD" \
    --J "$TKPERT_J" \
    --shape "$TKPER_SHAPE" \
    --lang "$LANG" \
    --data_path "$DATA_PATH" \
    --out_path "$OUT_PATH" \
    --chunk_size "$CHUNK_SIZE" \
    --save_num "$SAVE_NUM" \
    --input_format "$INPUT_FORMAT" \
    --data_domain "$DATA_DOMAIN"
