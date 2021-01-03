import fire
import logging
import os
from typing import List

import gensim.models

from utils.recipe_handling import (
    parse_raw_recipe,
    validate_recipe,
    predict_recipe,
)

from utils.regression import (
    load_regression,
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def predict_recipe_name(
    ingredients: List[str],
    embedding_model_path: str,
    regression_model_path: str,
):
    # load Word2Vec & regression models
    embedding_model = gensim.models.Word2Vec.load(embedding_model_path)
    regression_model = load_regression(regression_model_path)

    logging.info(f'ingredients include:')
    for ingredient in ingredients:
        logging.info(f'  {ingredient}')

    recipe_json = {
        'ingredients': ingredients,
    }
    recipe = parse_raw_recipe(recipe_json)

    if not validate_recipe(embedding_model, recipe):
        logging.info('Some ingredients are not in the Word2Vec model')
        return

    # predict recipe names
    prediction = predict_recipe(embedding_model, regression_model, recipe['ingredients'])
    logging.info(f'prediction:')
    for candidate in prediction:
        logging.info(f'  {candidate[0]} (similarity: {candidate[1]:.2f})')


if __name__ == '__main__':
    fire.Fire(predict_recipe_name)
