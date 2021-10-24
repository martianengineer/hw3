#!/usr/bin/env python3

from collections import OrderedDict
from typing import Iterable, List


class CountVectorizer:
    def __init__(self):
        self._vocabulary = OrderedDict()

    @staticmethod
    def _tokenize(doc: str):
        for word in doc.split():
            yield word.lower()

    def fit_transform(self, raw_documents: Iterable[str]) -> List[List[int]]:
        self._vocabulary = OrderedDict()
        for doc in raw_documents:
            for token in self._tokenize(doc):
                if token not in self._vocabulary:
                    self._vocabulary[token] = len(self._vocabulary)
        matrix = []
        for doc in raw_documents:
            matrix.append([0] * len(self._vocabulary))
            for token in self._tokenize(doc):
                matrix[-1][self._vocabulary[token]] += 1
        return matrix

    def get_feature_names(self) -> List[str]:
        return list(self._vocabulary.keys())
