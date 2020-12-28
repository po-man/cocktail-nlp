import fire
import json
import logging
import os
from typing import List


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def extract_phrases(recipes):
    phrases = []
    for recipe in recipes:
        phrases.append(recipe['name'])
        for ingredient in recipe['ingredients']:
            phrases.append(ingredient['ingredient'])
    phrases = [
        phrase.lower()
        for phrase in phrases
        if len(phrase.split('_')) > 1
    ]
    return phrases


def prepare_phrase_list(
    input_recipes_json_path: str,
    output_phrases_path: str,
):
    recipes = []
    with open(input_recipes_json_path, 'r') as json_file:
        recipes = json.load(json_file)

    phrases = extract_phrases(recipes)
    with open(output_phrases_path, 'w') as output_file:
        for phrase in phrases:
            output_file.write(f'{phrase}\n')


if __name__ == '__main__':
    fire.Fire(prepare_phrase_list)
