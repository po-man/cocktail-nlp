import fire
import json
import logging
import os

import gensim.models
import numpy as np

from utils.recipe_handling import (
    parse_raw_recipes,
    filter_recipes,
    predict_recipes,
)

from utils.regression import (
    load_regression,
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def evaluate(
    input_recipes_path: str,
    input_embedding_model_path: str,
    input_regression_model_path: str,
):
    # load Word2Vec & regression models
    embedding_model = gensim.models.Word2Vec.load(input_embedding_model_path)
    regression_model = load_regression(input_regression_model_path)

    # load recipes
    recipes_json = json.load(open(input_recipes_path, 'r'))
    logging.info(f'original #recipes: {len(recipes_json)}')

    # pre-process recipes
    recipes = parse_raw_recipes(recipes_json)
    recipes = filter_recipes(embedding_model, recipes)
    logging.info(f'filtered #recipes: {len(recipes)}')

    # predict recipe names
    predictions = predict_recipes(embedding_model, regression_model, recipes)

    # summarise
    num_recipes = len(predictions)
    top_1 = 0
    top_3 = 0
    for recipe, prediction in zip(recipes, predictions):
        if recipe['name'] == prediction[0][0]:
            top_1 += 1
        if recipe['name'] in [ name for name, score in prediction ]:
            top_3 += 1
    top_1_accuracy = top_1 / num_recipes
    top_3_accuracy = top_3 / num_recipes
    logging.info(f'Top-1 Accuracy: {top_1_accuracy} ({top_1}/{num_recipes})')
    logging.info(f'Top-3 Accuracy: {top_3_accuracy} ({top_3}/{num_recipes})')


if __name__ == '__main__':
    fire.Fire(evaluate)
