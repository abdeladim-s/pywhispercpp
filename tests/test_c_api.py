import unittest
import _pywhispercpp as pw


class TestCAPI(unittest.TestCase):

    model_file = '../whisper.cpp/models/for-tests-ggml-tiny.en.bin'
    ctx = None

    def test_whisper_init_from_file(self):
        ctx = pw.whisper_init_from_file(self.model_file)
        self.ctx = ctx
        return self.assertIsInstance(ctx, pw.whisper_context)

    def test_whisper_lang_str(self):
        return self.assertEqual(pw.whisper_lang_str(0), 'en')

    def test_whisper_lang_id(self):
        return self.assertEqual(pw.whisper_lang_id('en'), 0)

    def test_whisper_full_params(self):
        params = pw.whisper_full_params()
        return self.assertIsInstance(params.n_threads, int)

    def __del__(self):
        pw.whisper_free(self.ctx)


if __name__ == '__main__':
    unittest.main()
