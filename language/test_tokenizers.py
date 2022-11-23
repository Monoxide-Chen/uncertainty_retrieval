import unittest

from language import tokenizers
from language.abc import _BOS_TOKEN, _EOS_TOKEN, _PAD_TOKEN


class TestBasicTokenizer(unittest.TestCase):
    def test_basic_tokenizers_make_text_lowercase_and_add_start_end_tokens_when_tokenizing(self):
        tokenizer = tokenizers.BasicTokenizer()
        self.assertListEqual([_BOS_TOKEN, 'hi', 'bye', 'mama', _EOS_TOKEN], tokenizer.tokenize('hi, bye, mama!'))

    def test_basic_tokenizers_remove_start_end_tokens_and_padding_when_detokenizing(self):
        tokenizer = tokenizers.BasicTokenizer()
        tokens = ['1', '2', _BOS_TOKEN, 'hi', _PAD_TOKEN, 'bye', 'mama', _PAD_TOKEN, _EOS_TOKEN, '3', '4']
        self.assertEqual('hi bye mama', tokenizer.detokenize(tokens))


class TestNltkTokenizer(unittest.TestCase):
    def test_nltk_tokenizers_make_text_lowercase_and_add_start_end_tokens_when_tokenizing(self):
        tokenizer = tokenizers.NltkTokenizer()
        self.assertListEqual([_BOS_TOKEN, 'hi', ',', 'bye', ',', 'mama', '!', _EOS_TOKEN],
                             tokenizer.tokenize('hi, bye, Mama!'))

    def test_basic_tokenizers_remove_start_end_tokens_and_padding_when_detokenizing(self):
        tokenizer = tokenizers.NltkTokenizer()
        tokens = ['1', '2', _BOS_TOKEN, 'hi', _PAD_TOKEN, 'bye', 'mama', _PAD_TOKEN, _EOS_TOKEN, '3', '4']
        self.assertEqual('hi bye mama', tokenizer.detokenize(tokens))


if __name__ == '__main__':
    unittest.main()
