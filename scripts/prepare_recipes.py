import fire
import json
import logging
import os
import re
from typing import List

from bs4 import BeautifulSoup
from ingreedypy import Ingreedy


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

ingredient_parser = Ingreedy()


def validate_ingredient(ingredient) -> bool:
    if ingredient['ingredient'] is None:
        return False
    for quantity in ingredient['quantity']:
        if quantity['amount'] is None:
            return False
    return True


def validate_recipe(recipe) -> bool:
    return len(recipe['ingredients']) > 1


def rename(name):
    name = re.sub('[^\w\s]', '', name.lower())
    name = name.replace(' ', '_')
    return name


def parse_ingredient(ingredient_str):
    ingredient = ingredient_parser.parse(ingredient_str)
    if ingredient['ingredient']:
        ingredient['ingredient'] = rename(ingredient['ingredient'])
    return ingredient


def parse_recipe(
    recipe_div,
):
    name = recipe_div.h3.get_text()
    img_src = recipe_div.img['src']
    img_file = img_src.split('/')[-1]
    recipe_paragraphs = recipe_div \
        .find('div', class_='content-appear') \
        .find_all('p')
    ingredient_strs = recipe_paragraphs[0].get_text('\n').split('\n')
    ingredients = [
        parse_ingredient(ingredient_str)
        for ingredient_str in ingredient_strs
    ]
    description = recipe_paragraphs[1].get_text()
    return {
        'name': rename(name),
        'img_file': img_file,
        'ingredients': [
            ingredient
            for ingredient in ingredients
            if validate_ingredient(ingredient)
        ],
        'description': description,
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
    recipes_json = [
        recipe
        for recipe in recipes_json
        if validate_recipe(recipe)
    ]

    with open(output_recipes_json_path, 'w') as json_file:
        json.dump(recipes_json, json_file)


if __name__ == '__main__':
    fire.Fire(prepare_recipes)
