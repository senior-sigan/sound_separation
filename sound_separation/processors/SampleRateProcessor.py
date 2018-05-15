# -*- coding: utf-8 -*-
from librosa.core import resample

from sound_separation.processors.IProcessor import IProcessor


class SampleRateProcessor(IProcessor):
    def __init__(self, sample_rate=16000, origin_sample_rate=44100):
        self.origin_sample_rate = origin_sample_rate
        self.sample_rate = sample_rate

    def process(self, data):
        return resample(data, orig_sr=self.origin_sample_rate, target_sr=self.sample_rate)
