#!/bin/bash

DATA_DIR='/__data__'

# Data preparation (recipes)

mkdir -p "$DATA_DIR/recipes"
curl -X GET https://www.socialandcocktail.co.uk/top-100-cocktails/ \
    > "$DATA_DIR/recipes/top-100-cocktails.html"

python prepare_recipes.py \
    --input_html_path="$DATA_DIR/recipes/top-100-cocktails.html" \
    --output_recipes_json_path="$DATA_DIR/recipes.json"

python prepare_phrase_list.py \
    --input_recipes_json_path="$DATA_DIR/recipes.json" \
    --output_phrases_path="$DATA_DIR/phrases.txt"


# Data preparation (corpus)

rm -f "$DATA_DIR/corpus.txt"
python prepare_corpus.py \
    --input_ebook_paths="$DATA_DIR/ebooks/**/*.epub" \
    --output_corpus_path="$DATA_DIR/corpus.txt"

python prepare_corpus_with_phrases.py \
    --input_phrases_path="$DATA_DIR/phrases.txt" \
    --input_corpus_path="$DATA_DIR/corpus.txt" \
    --output_corpus_path="$DATA_DIR/corpus_with_phrases.txt"


# Model training

python train_embedding_model.py \
    --input_corpus_path="$DATA_DIR/corpus_with_phrases.txt" \
    --output_model_path="$DATA_DIR/embedding_model.bin" \
    --epochs=10 \
    --vector_size=100

python train_regression.py \
    --input_embedding_model_path="$DATA_DIR/embedding_model.bin" \
    --input_recipes_path="$DATA_DIR/recipes.json" \
    --output_regression_model_path="$DATA_DIR/regression_model.bin"


# Evaluation

python evaluate.py \
    --input_recipes_path="$DATA_DIR/recipes.json" \
    --input_embedding_model_path="$DATA_DIR/embedding_model.bin" \
    --input_regression_model_path="$DATA_DIR/regression_model.bin"

python predict_recipe_name.py \
    --ingredients='[
        "50 ml White Rum",
        "2 Mint leaves",
        "10 ml Sugar Syrup",
        "25 ml Lime Juice",
    ]' \
    --embedding_model_path="$DATA_DIR/embedding_model.bin" \
    --regression_model_path="$DATA_DIR/regression_model.bin"
