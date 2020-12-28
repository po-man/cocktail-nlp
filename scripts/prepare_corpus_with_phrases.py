import fire
import json
import logging
import os
from typing import List


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))



def prepare_corpus_with_phrases(
    input_phrases_path: str,
    input_corpus_path: str,
    output_corpus_path: str,
):
    phrases = [
        line.replace('\n', '')
        for line in open(input_phrases_path, 'r').readlines()
    ]
    phrase_map = [
        (phrase.replace('_', ' '), phrase)
        for phrase in phrases
    ]
    with open(input_corpus_path, 'r') as input_corpus_file, \
            open(output_corpus_path, 'w') as output_corpus_file:
        for input_corpus in input_corpus_file.readlines():
            output_corpus = input_corpus
            for from_phrase, to_phrase in phrase_map:
                output_corpus = output_corpus.replace(from_phrase, to_phrase)
            output_corpus_file.write(output_corpus)


if __name__ == '__main__':
    fire.Fire(prepare_corpus_with_phrases)
