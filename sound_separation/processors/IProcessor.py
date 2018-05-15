# -*- coding: utf-8 -*-

from abc import abstractmethod


class IProcessor:
    @abstractmethod
    def process(self, data):
        pass
