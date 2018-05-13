# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

import pandas as pd
import soundfile as sf
import stempeg
import tqdm


def split_files(input_path, output_path, output_csv_path):
    df = []
    sample_len = 3  # seconds

    files = glob(os.path.join(input_path, "*.mp4"))
    files = files[:1]
    for file in files:
        audio, rate = stempeg.read_stems(file)
        duration = audio.shape[1]
        samples = int(duration / (sample_len * rate))
        for stem in range(1, 5):
            for n in range(samples - 2):
                start = n * sample_len
                end = (n + 1) * sample_len
                sample = audio[stem, start * rate:end * rate, 0]
                fname = os.path.basename(file)
                sample_name = "{}_{}-{}_{}.wav".format(fname, start, end, stem)
                sample_path = os.path.join(output_path, sample_name)
                sf.write(sample_path, sample, rate)
                df.append({"fname": sample_name, "start": start, "end": end, "category": stem, "origin_fname": fname})

    df = pd.DataFrame(df)
    df.to_csv(output_csv_path)


if __name__ == '__main__':
    split_files(sys.argv[1], sys.argv[2], sys.argv[3])
