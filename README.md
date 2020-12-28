# Cocktail-NLP

In the project, we try to predict cocktail names given the portions of ingredients.

It consists of 2 main stages:
- Word2Vec model trained from cocktail-related corpus
- Regression model mapping ingredient portions to cocktail names

## Steps

1. Data preparation
  - Corpus (from EPUB ebooks)
  - Recipes (from [Top-100 Cocktails][1])

2. Model training
  - Word2vec from corpus
  - Regression from recipes

3. Evaluation


[1]: https://www.socialandcocktail.co.uk/top-100-cocktails/
