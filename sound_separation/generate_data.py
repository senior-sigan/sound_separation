# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import soundfile as sf
import stempeg
from tqdm import tqdm


class DataReader:
    def __init__(self, input_path, output_path, output_csv_path, n_jobs=4):
        self.n_jobs = n_jobs
        self.output_csv_path = output_csv_path
        self.output_path = output_path
        self.input_path = input_path
        self.sample_len = 3

    def read_and_save(self):
        os.makedirs(self.output_path)
        files = glob(os.path.join(self.input_path, "*.mp4"))
        df = []
        for file in tqdm(files):
            for el in self._process(file):
                df.append(el)
        df = pd.DataFrame(df)
        df.to_csv(self.output_csv_path)

    def _process(self, file):
        audio, rate = stempeg.read_stems(file)
        duration = audio.shape[1]
        samples = int(duration / (self.sample_len * rate))
        for stem in range(1, 5):
            for n in range(samples - 2):
                start = n * self.sample_len
                end = (n + 1) * self.sample_len
                sample = audio[stem, start * rate:end * rate, 0]
                fname = os.path.basename(file)
                sample_name = "{}_{}-{}_{}.wav".format(fname, start, end, stem)
                sample_path = os.path.join(self.output_path, sample_name)
                sf.write(sample_path, sample, rate)
                yield {
                    "fname": sample_name,
                    "start": start,
                    "end": end,
                    "category": stem,
                    "origin_fname": fname,
                    "is_silence": self._is_silence(sample)
                }

    def _is_silence(self, sample):
        return np.median(np.abs(sample)) < 0.0001 or np.max(np.abs(sample)) < 0.00001


if __name__ == '__main__':
    reader = DataReader(sys.argv[1], sys.argv[2], sys.argv[3])
    reader.read_and_save()
