# -*- coding: utf-8 -*-
from sound_separation.processors.EmphasisProcessor import EmphasisProcessor
from sound_separation.processors.LoadProcessor import LoadProcessor
from sound_separation.processors.SampleRateProcessor import SampleRateProcessor
from sound_separation.processors.SpectralProcessor import SpectralProcessor


class SoundChain:
    def __init__(self, config):
        self.processors = [
            LoadProcessor(config['root_path']),
            EmphasisProcessor(),
            SampleRateProcessor(),
            SpectralProcessor()
        ]

    def run(self, file_name):
        data = file_name
        for proc in self.processors:
            data = proc.process(data)
        return data
