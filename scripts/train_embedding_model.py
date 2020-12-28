import fire
import logging
import os

import gensim.models

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def train_embedding_model(
    input_corpus_path: str,
    output_model_path: str,
    epochs: int = 10,
    vector_size: int = 100,
):
    total_examples = len(open(input_corpus_path, 'r').readlines())
    model = gensim.models.Word2Vec(size=vector_size)
    sentences = gensim.models.word2vec.LineSentence(input_corpus_path)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=total_examples, epochs=epochs)
    model.save(output_model_path)


if __name__ == '__main__':
    fire.Fire(train_embedding_model)
