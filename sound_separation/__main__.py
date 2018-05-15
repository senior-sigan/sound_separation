# -*- coding: utf-8 -*-

import numpy as np

from sound_separation.SoundChain import SoundChain
from sound_separation.consts import LABELS
from sound_separation.data import load_data, train_generator, valid_generator
from sound_separation.models.resnet import ResNetClassifier


def main():
    model = ResNetClassifier(input_shape=(257, 301, 2), labels=LABELS)
    chain = SoundChain({'root_path': '/home/ilya/Data/musdb18/split_train/'})
    train_df, valid_df = load_data('/home/ilya/Data/musdb18/data.csv')
    batch_size = 32
    sample_size = 2000
    train_gen = train_generator(train_df, batch_size, chain, n=sample_size)
    valid_gen = valid_generator(valid_df, batch_size, chain)
    model.train(train_gen, valid_gen, {
        'steps_per_epoch': sample_size * len(LABELS) / batch_size,
        'epochs': 200,
        'validation_steps': int(np.ceil(valid_df.shape[0] / batch_size)),
        'tensorboard_dir': '/home/ilya/Data/musdb18/output/tensorboard',
        'batch_size': 32,
        'chekpoints_path': '/home/ilya/Data/musdb18/output/models'
    })


if __name__ == '__main__':
    main()
