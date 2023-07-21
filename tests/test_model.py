#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test model.py
"""
import unittest
from unittest import TestCase

from pywhispercpp.model import Model, Segment

if __name__ == '__main__':
    pass


class TestModel(TestCase):
    audio_file = '../whisper.cpp/samples/jfk.wav'
    model = Model("tiny")

    def test_transcribe(self):
        segments = self.model.transcribe(self.audio_file)
        return self.assertIsInstance(segments, list) and \
               self.assertIsInstance(segments[0], Segment) if len(segments) > 0 else True

    def test_get_params(self):
        params = self.model.get_params()
        return self.assertIsInstance(params, dict)

    def test_lang_max_id(self):
        n = self.model.lang_max_id()
        return self.assertGreater(n, 0)

    def test_available_languages(self):
        av_langs = self.model.available_languages()
        return self.assertIsInstance(av_langs, list) and self.assertGreater(len(av_langs), 1)

    def test__load_audio(self):
        audio_arr = self.model._load_audio(self.audio_file)
        return self.assertIsNotNone(audio_arr)


if __name__ == '__main__':
    unittest.main()
