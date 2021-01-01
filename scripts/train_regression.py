import fire
import json
import logging
import os

import gensim.models

from utils.recipe_handling import (
    filter_recipes,
    encode_recipes,
    prepare_matrices,
)

from utils.regression import (
    fit_regression,
    save_regression,
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def train_regression(
    input_embedding_model_path: str,
    input_recipes_path: str,
    output_regression_model_path: str,
):
    # load Word2Vec model
    embedding_model = gensim.models.Word2Vec.load(input_embedding_model_path)

    # load recipes
    recipes = json.load(open(input_recipes_path, 'r'))
    logging.info(f'original #recipes: {len(recipes)}')

    # pre-process recipes
    recipes = filter_recipes(embedding_model, recipes)
    logging.info(f'filtered #recipes: {len(recipes)}')
    recipes = encode_recipes(embedding_model, recipes)
    source_matrix, target_matrix = prepare_matrices(recipes)

    # fit regression model
    regression_model = fit_regression(source_matrix, target_matrix)

    # save regression model
    save_regression(regression_model, output_regression_model_path)


if __name__ == '__main__':
    fire.Fire(train_regression)
