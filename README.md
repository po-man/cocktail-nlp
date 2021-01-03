# Cocktail-NLP

In the project, we try to predict cocktail names given the portions of ingredients.

Example:
```
input ingredients:
  - 50 ml White Rum
  - 2 Mint leaves
  - 10 ml Sugar Syrup
  - 25 ml Lime Juice
output recipe name:
  - mojito (similarity: 0.98)
```

It consists of 2 main stages:
- Word2Vec model trained from cocktail-related corpus
- Regression model mapping ingredient portions to cocktail names


## Word2Vec model

Corpus is prepared from ~200 cocktail/wine-related e-books (~4GB) (not included in this repository).

Word2Vec model is trained from the prepared corpus with [Gensim][1]


## Formulation of Regression

The first stage (Word2Vec) model converts vocabularies into vectors, but has no information about the ingredients-recipe-name relationship.

The second stage finds a linear transformation (from vocab-space into vocab-space) to minimise the following loss:

![\sum_i^n \big(w_i X(R_i) - X(I_i)\big)^2](https://latex.codecogs.com/gif.latex?\sum_i^n&space;\big(w_i&space;X(R_i)&space;-&space;X(I_i)\big)^2)

where
- ![X(.)](https://latex.codecogs.com/gif.latex?X(.)) is the vocab-space-to-vocab-space transformation,
- ![R_i](https://latex.codecogs.com/gif.latex?R_i) is the vector of target recipe name,
- ![I_i](https://latex.codecogs.com/gif.latex?I_i) is the vector of source ingredient,
- ![w_i](https://latex.codecogs.com/gif.latex?w_i) is the normalised weight of the corresponding ingredient in the recipe.


## Large files

Large files can be found here:
- [recipes.json](https://drive.google.com/file/d/1duVMEsYtVkne_n5pzih96MW69Oq4LWxr/view?usp=sharing)
- [corpus.txt](https://drive.google.com/file/d/1uTC6QqRZUc8NMnnh3RLYa437mgh_iuC6/view?usp=sharing)
- [corpus_with_phrases.txt](https://drive.google.com/file/d/1mhZ3R3VEUc7cuQb8wZV8aZprc6MQiIk4/view?usp=sharing)
- [phrases.txt](https://drive.google.com/file/d/1zIVYFl_bYYvLvh3bhMY1jVIH5_ddlXCK/view?usp=sharing)
- [embedding_model.bin](https://drive.google.com/file/d/1yEhAvUTQZVGlqaUf1KOP00r2yOm-gDDM/view?usp=sharing)
- [regression_model.bin](https://drive.google.com/file/d/17Dn_FkpiQcWziiLOj9bAVvMtgVDjobs8/view?usp=sharing)

Put them under `/__data__` then you may play with the `run.sh` script.


[1]: https://radimrehurek.com/gensim/
[2]: https://www.socialandcocktail.co.uk/top-100-cocktails/
