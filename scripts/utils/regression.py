import json
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_regression(source_matrix, target_matrix):
    print(f'source_matrix: {source_matrix.shape}')
    print(f'target_matrix: {target_matrix.shape}')
    model = LinearRegression().fit(source_matrix, target_matrix)
    score = model.score(source_matrix, target_matrix)
    print(f'score: {score}')
    return model


def eval_regression(model, input_vector):
    return np.squeeze(model.predict(np.expand_dims(input_vector, axis=0)))


def save_regression(model, filepath):
    pickle.dump(model, open(filepath, 'wb'))


def load_regression(filepath):
    model = pickle.load(open(filepath, 'rb'))
    return model


def fit_regression_from_recipes(recipes):
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

    model = fit_regression(source_matrix, target_matrix);
    return model


def eval_regression_on_recipe(model, recipe):
    vector_size = recipe['vector'].shape[0]
    num_ingredients = sum([ 1
        for ingredient in recipe['ingredients']
    ])
    source_matrix = np.zeros((1 + num_ingredients, vector_size))
    prediction = model.predict(source_matrix)
    print(prediction)


