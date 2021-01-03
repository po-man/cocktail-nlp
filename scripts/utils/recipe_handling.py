import json
import re

import numpy as np
from ingreedypy import Ingreedy

from .regression import eval_regression


def parse_raw_recipe(raw_recipe):
    def normalise_name(name):
        name = re.sub('[^\w\s]', '', name.lower())
        name = name.replace(' ', '_')
        return name

    recipe = {}
    if 'name' in raw_recipe.keys():
        recipe['name'] = normalise_name(raw_recipe['name'])

    ingredient_parser = Ingreedy()
    ingredients = []
    for ingredient_str in raw_recipe['ingredients']:
        ingredient = ingredient_parser.parse(ingredient_str)
        if ingredient['ingredient']:
            ingredient['ingredient'] = normalise_name(ingredient['ingredient'])
            if all([
                quantity['amount'] is not None
                for quantity in ingredient['quantity']
            ]):
                ingredients.append(ingredient)
    recipe['ingredients'] = ingredients
    return recipe


def parse_raw_recipes(raw_recipes):
    recipes = [
        parse_raw_recipe(raw_recipe)
        for raw_recipe in raw_recipes
    ]
    return [
        recipe
        for recipe in recipes
        if len(recipe['ingredients']) > 1
    ]


def validate_recipe(embedding_model, recipe) -> bool:
    if 'name' in recipe.keys():
        if name not in embedding_model.wv.vocab:
            return False
    for ingredient in recipe['ingredients']:
        ingredient_name = ingredient['ingredient']
        if ingredient_name not in embedding_model.wv.vocab:
            return False
    return True


def filter_recipes(embedding_model, recipes):
    return [
        recipe
        for recipe in recipes
        if validate_recipe(embedding_model, recipe)
    ]


def encode_ingredients(embedding_model, ingredients):
    def quantity_to_weight(quantity) -> float:
        if len(quantity) == 0:
            return None

        quan = quantity[0]
        factor = 1
        if quan['unit_type'] == 'metric':
            if quan['unit'] == 'milliliter':
                factor = 1
        elif quan['unit_type'] == 'english':
            if quan['unit'] == 'teaspoon':
                factor = 4.9289
        elif quan['unit_type'] == 'imprecise':
            if quan['unit'] == 'dash':
                factor = 0.92
            elif quan['unit'] == 'pinch':
                factor = 0.31
        else:
            return None
        return quan['amount'] * factor

    def adjust_weights(weights):
        total_weight_with_unit = sum([
            weight
            for weight in weights
            if weight is not None
        ])
        new_weights = [
            weight / total_weight_with_unit if weight is not None else 1 / len(weights)
            for weight in weights
        ]
        return new_weights

    def encode_ingredient(embedding_model, ingredient):
        ingredient['weight'] = quantity_to_weight(ingredient['quantity'])
        ingredient['vector'] = embedding_model.wv[ingredient['ingredient']]
        return ingredient

    ingredients = [
        encode_ingredient(embedding_model, ingredient)
        for ingredient in ingredients
    ]

    weights = [
        ingredient['weight']
        for ingredient in ingredients
    ]
    new_weights = adjust_weights(weights)
    for idx, ingredient in enumerate(ingredients):
        ingredients[idx]['weight'] = new_weights[idx]

    return ingredients


def encode_recipe(embedding_model, recipe):
    recipe['vector'] = embedding_model.wv[recipe['name']]
    recipe['ingredients'] = encode_ingredients(embedding_model, recipe['ingredients'])
    return recipe


def encode_recipes(embedding_model, recipes):
    return [
        encode_recipe(embedding_model, recipe)
        for recipe in recipes
    ]


def prepare_matrices(recipes):
    vector_size = recipes[0]['vector'].shape[0]
    num_ingredients = sum([ 1
        for recipe in recipes
        for ingredient in recipe['ingredients']
    ])
    print(f'vector_size: {vector_size}')
    print(f'len(recipes): {len(recipes)}')
    print(f'num_ingredients: {num_ingredients}')
    source_matrix = np.zeros((len(recipes) + num_ingredients, vector_size))
    target_matrix = np.zeros((len(recipes) + num_ingredients, vector_size))
    idx = 0
    for recipe in recipes:
        source_matrix[idx,:] = recipe['vector']
        target_matrix[idx,:] = recipe['vector']
        idx = idx + 1
        for ingredient in recipe['ingredients']:
            source_matrix[idx,:] = ingredient['vector']
            target_matrix[idx,:] = recipe['vector'] * ingredient['weight']
            idx = idx + 1
    return (source_matrix, target_matrix)


def predict_recipe(
    embedding_model,
    regression_model,
    ingredients,
    top_k = 3,
):
    ingredients = encode_ingredients(embedding_model, ingredients)
    ingredients_vectors = np.array([
        eval_regression(
            regression_model,
            ingredient['vector'],
        ) * ingredient['weight']
        for ingredient in ingredients
    ])
    prediction_vector = np.sum(ingredients_vectors, axis=0)
    prediction = embedding_model.wv.most_similar([prediction_vector], topn=top_k)
    return prediction


def predict_recipes(
    embedding_model,
    regression_model,
    recipes,
):
    predictions = [
        predict_recipe(
            embedding_model,
            regression_model,
            recipe['ingredients']
        )
        for recipe in recipes
    ]
    return predictions
