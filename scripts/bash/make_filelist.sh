#!/bin/bash

# Base directories
BASE_DIR=$(git rev-parse --show-toplevel)
BASE_DIR_DEV="$BASE_DIR/DCASE_2023_Challenge_Task_7_Dataset/dev"
BASE_DIR_EVAL="$BASE_DIR/DCASE_2023_Challenge_Task_7_Dataset/eval"

# Generate file list for training
find "$BASE_DIR_DEV" -type f > "$BASE_DIR/DCASE_2023_Challenge_Task_7_Dataset/train.txt"

# Generate file list for evaluation
find "$BASE_DIR_EVAL" -type f > "$BASE_DIR/DCASE_2023_Challenge_Task_7_Dataset/eval.txt"
echo "Filelists are created in the dataset directory."