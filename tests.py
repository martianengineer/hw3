#!/usr/bin/env python3

import unittest
from text_vectorizer import CountVectorizer


class TestCountVectorizerMethods(unittest.TestCase):
    def test_empty_corpus(self):
        corpus = []
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        self.assertEqual(matrix, [])
        self.assertEqual(feature_names, [])

    def test_empty_texts_corpus(self):
        corpus = ["", ""]
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        self.assertEqual(matrix, [[], []])
        self.assertEqual(feature_names, [])

    def test_corpus(self):
        corpus = ["Crock Pot Pasta Never boil pasta again",
                  "Pasta Pomodoro Fresh ingredients Parmesan to taste"]
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        expected_matrix = [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
        self.assertEqual(matrix, expected_matrix)
        expected_feature_names = ['crock', 'pot', 'pasta', 'never', 'boil',
                                  'again', 'pomodoro', 'fresh', 'ingredients',
                                  'parmesan', 'to', 'taste']
        self.assertEqual(feature_names, expected_feature_names)


if __name__ == '__main__':
    unittest.main()
