import fire
import json
import logging
import os
from pprint import pprint

import gensim.models
import numpy as np

from utils.recipe_handling import (
    filter_recipes,
    encode_ingredients,
)

from utils.regression import (
    load_regression,
    eval_regression,
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


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



def evaluate(
    input_recipes_path: str,
    input_embedding_model_path: str,
    input_regression_model_path: str,
):
    # load Word2Vec & regression models
    embedding_model = gensim.models.Word2Vec.load(input_embedding_model_path)
    regression_model = load_regression(input_regression_model_path)

    # load recipes
    recipes = json.load(open(input_recipes_path, 'r'))
    logging.info(f'original #recipes: {len(recipes)}')

    # pre-process recipes
    recipes = filter_recipes(embedding_model, recipes)
    logging.info(f'filtered #recipes: {len(recipes)}')

    #recipe = recipes[0]
    predictions = predict_recipes(embedding_model, regression_model, recipes)

    comparisons = [
        {
            'prediction': prediction,
            'ground_truth': recipe['name']
        }
        for recipe, prediction in zip(recipes, predictions)
    ]
    pprint(comparisons)


if __name__ == '__main__':
    fire.Fire(evaluate)
