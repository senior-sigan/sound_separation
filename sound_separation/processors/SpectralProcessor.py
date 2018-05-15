# -*- coding: utf-8 -*-

import numpy as np
from librosa.core import stft

from sound_separation.processors.IProcessor import IProcessor


class SpectralProcessor(IProcessor):
    def process(self, data):
        """
        Returns 3-d matrix of sizes [257,301,2]
        :param data:
        :return:
        """
        spectr = stft(data, n_fft=512, hop_length=160)
        return np.concatenate((spectr.real[:, :, np.newaxis], spectr.imag[:, :, np.newaxis]), axis=2)
