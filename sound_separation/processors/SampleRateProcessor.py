# -*- coding: utf-8 -*-
from sound_separation.processors.IProcessor import IProcessor


class SampleRateProcessor(IProcessor):
    def __init__(self, files_path, sample_rate):
        self.sample_rate = sample_rate
        self.files_path = files_path
    
    def process(self, fname):
        pass
