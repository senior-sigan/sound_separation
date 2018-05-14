# -*- coding: utf-8 -*-
from abc import abstractmethod


class Classifier:
    def __init__(self, input_shape, labels) -> None:
        self.input_shape = input_shape
        self.labels = labels
        self._name = "AbstractClassifier"

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def train(self, train_gen, validation_gen, params):
        pass
