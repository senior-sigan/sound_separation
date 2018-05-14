# -*- coding: utf-8 -*-
from sound_separation.processors.IProcessor import IProcessor


class SpectralProcessor(IProcessor):
    def __init__(self, files_path):
        self.files_path = files_path

    def process(self, fname):
        # TODO: read file, build spectrum
        pass
