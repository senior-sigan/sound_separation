# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd
from keras.utils import to_categorical

from sound_separation.SoundChain import SoundChain
from sound_separation.consts import LABELS


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['category'] = df['category'] - 1
    files = df['origin_fname'].unique()
    train = df[df['origin_fname'].isin(files[0:60])]
    test = df[df['origin_fname'].isin(files[60:])]

    return train[['category', 'fname']], test[['category', 'fname']]


def train_generator(df: pd.DataFrame, batch_size: int, sound_chain: SoundChain, n=2000):
    while True:
        this_train = df.groupby('category').apply(sampling(n))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), batch_size):
            end = min(start + batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]

            x_batch = [sound_chain.run(this_train['fname'].values[i]) for i in i_train_batch]
            y_batch = [this_train['category'].values[i] for i in i_train_batch]

            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes=len(LABELS))
            yield x_batch, y_batch


def valid_generator(df, batch_size, sound_chain: SoundChain, with_y=True):
    while True:
        ids = list(range(df.shape[0]))
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            i_val_batch = ids[start:end]

            x_batch = [sound_chain.run(df['fname'].values[i]) for i in i_val_batch]
            y_batch = [df['category'].values[i] for i in i_val_batch]

            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes=len(LABELS))

            if with_y:
                yield x_batch, y_batch
            else:
                yield x_batch


def sampling(n):
    """
    Pandas dataframe can return sample of a subset.
    But this function can create extra duplications, so subset could be extracted
    :param n: subset size
    :return: sampling function
    """

    def _sample(x):
        if n > x.shape[0]:
            # generate dups
            count = n // x.shape[0] + 1
            x = pd.concat([x] * count)
            return x.sample(n=n)
        else:
            return x.sample(n=n)

    return _sample
