import _pywhispercpp as pw

import unittest
from unittest import TestCase


class TestCAPI(TestCase):

    model_file = './whisper.cpp/models/for-tests-ggml-tiny.en.bin'

    def test_whisper_init_from_file(self):
        ctx = pw.whisper_init_from_file(self.model_file)
        self.assertIsInstance(ctx, pw.whisper_context)

    def test_whisper_lang_str(self):
        return self.assertEqual(pw.whisper_lang_str(0), 'en')

    def test_whisper_lang_id(self):
        return self.assertEqual(pw.whisper_lang_id('en'), 0)

    def test_whisper_full_params(self):
        params = pw.whisper_full_params()
        return self.assertIsInstance(params.n_threads, int)

    def test_initial_prompt_param(self):
        params = pw.whisper_full_params()
        prompt = "A" + " test"
        params.initial_prompt = prompt
        del prompt
        return self.assertEqual("A test", params.initial_prompt)

    def test_language_param(self):
        whisper_params = pw.whisper_full_params()
        whisper_params.language = "en"  # This will invoke the problematic setter
        temp_string = "es"
        whisper_params.language = temp_string  # Invoke the problematic setter again
        del temp_string
        return self.assertEqual("es", whisper_params.language)


if __name__ == '__main__':
    unittest.main()
