import fire
import glob
import logging
import os
import re
from typing import List

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def html_to_corpus(
    html: str,
    min_sentence_length: int,
    min_vocab_length: int,
) -> List[str]:
    soup = BeautifulSoup(html)
    sentences = [
        [
            vocab.lower()
            for vocab in re.sub('[^\w-]', ' ', text).split()
            if len(vocab) >= min_vocab_length
        ]
        for p in soup.find_all('p')
        for text in p.get_text('\n').split('\n')
    ]
    sentences = [
        ' '.join(vocabs)
        for vocabs in sentences
        if len(vocabs) >= min_sentence_length
    ]
    return sentences


def ebook_to_corpus(ebook, config: dict) -> List:
    sentences = [
        sentence
        for doc in ebook.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        for sentence in html_to_corpus(
            doc.content,
            config['min_sentence_length'],
            config['min_vocab_length'],
        )
    ]
    return sentences


def parse_ebook(ebook_path: str, corpus_path: str, config: dict):
    with open(corpus_path, 'a') as corpus_file:
        ebook = epub.read_epub(ebook_path)
        corpus_text = ebook_to_corpus(ebook, config)
        for sentence in corpus_text:
            corpus_file.write(f'{sentence}\n')


def prepare_corpus(
    input_ebook_paths: str,
    output_corpus_path: str,
    min_sentence_length: int = 10,
    min_vocab_length: int = 2,
):
    config = {
        'min_sentence_length': min_sentence_length,
        'min_vocab_length': min_vocab_length,
    }
    for ebook_path in glob.glob(input_ebook_paths):
        try:
            parse_ebook(ebook_path, output_corpus_path, config)
            logging.info(f'Parsed: {ebook_path}')
        except:
            logging.error(f'Failed to parse: {ebook_path}')


if __name__ == '__main__':
    fire.Fire(prepare_corpus)
