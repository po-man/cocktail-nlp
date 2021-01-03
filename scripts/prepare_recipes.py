import fire
import json
import logging
import os
import re
from typing import List

from bs4 import BeautifulSoup


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def parse_recipe(
    recipe_div,
):
    name = recipe_div.h3.get_text()
    recipe_paragraphs = recipe_div \
        .find('div', class_='content-appear') \
        .find_all('p')
    ingredients = re.split(' \/ |\n|,', recipe_paragraphs[0].get_text('\n'))
    return {
        'name': name,
        'ingredients': ingredients
    }


def prepare_recipes(
    input_html_path: str,
    output_recipes_json_path: str,
):
    html = open(input_html_path).read()
    soup = BeautifulSoup(html, features='html.parser')
    recipe_divs = soup \
        .find(id='content-home') \
        .section \
        .find_all('div', class_='recipe_summary pjax')
    recipes_json = [
        parse_recipe(recipe_div)
        for recipe_div in recipe_divs
    ]

    with open(output_recipes_json_path, 'w') as json_file:
        json.dump(recipes_json, json_file)


if __name__ == '__main__':
    fire.Fire(prepare_recipes)
