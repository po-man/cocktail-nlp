#!/bin/bash


# Data preparation (recipes)

mkdir -p '/__data__/recipes'
curl -X GET https://www.socialandcocktail.co.uk/top-100-cocktails/ \
    > '/__data__/recipes/top-100-cocktails.html'

python prepare_recipes.py \
    --input_html_path='/__data__/recipes/top-100-cocktails.html' \
    --output_recipes_json_path='/__data__/recipes.json'

python prepare_phrase_list.py \
    --input_recipes_json_path='/__data__/recipes.json' \
    --output_phrases_path='/__data__/phrases.txt'


# Data preparation (corpus)

rm -f '/__data__/corpus.txt'
python prepare_corpus.py \
    --input_ebook_paths='/__data__/ebooks/**/*.epub' \
    --output_corpus_path='/__data__/corpus.txt'

python prepare_corpus_with_phrases.py \
    --input_phrases_path='/__data__/phrases.txt' \
    --input_corpus_path='/__data__/corpus.txt' \
    --output_corpus_path='/__data__/corpus_with_phrases.txt'


# Model training (Word2Vec)

python train_embedding_model.py \
    --input_corpus_path='/__data__/corpus_with_phrases.txt' \
    --output_model_path='/__data__/embedding_model.bin' \
    --epochs=10 \
    --vector_size=100