import json

import numpy as np


def filter_recipes(embedding_model, recipes):
    def validate_recipe(embedding_model, recipe) -> bool:
        name = recipe['name'].lower()
        if name not in embedding_model.wv.vocab:
            return False
        for ingredient in recipe['ingredients']:
            ingredient_name = ingredient['ingredient'].lower()
            if ingredient_name not in embedding_model.wv.vocab:
                return False
        return True

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
