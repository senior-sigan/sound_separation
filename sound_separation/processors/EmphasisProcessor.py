# -*- coding: utf-8 -*-
import numpy as np

from sound_separation.processors.IProcessor import IProcessor


class EmphasisProcessor(IProcessor):
    def __init__(self, alpha=0.97):
        self.alpha = alpha

    def process(self, data):
        return np.append(data[0], data[1:] - self.alpha * data[:-1])
