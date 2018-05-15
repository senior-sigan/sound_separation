# -*- coding: utf-8 -*-
import os

from librosa.core import load

from sound_separation.processors.IProcessor import IProcessor


class LoadProcessor(IProcessor):
    def __init__(self, root_path, sample_rate=44100):
        self.sample_rate = sample_rate
        self.root_path = root_path

    def process(self, fname):
        path = os.path.join(self.root_path, fname)
        wav, _ = load(path, sr=self.sample_rate)
        return wav
