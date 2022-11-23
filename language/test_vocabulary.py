import unittest

from language.tokenizers import BasicTokenizer
from language.vocabulary import SimpleVocabulary


class TestSimpleVocabulary(unittest.TestCase):
    def setUp(self):
        self.vocabulary = SimpleVocabulary(BasicTokenizer())
        self.vocabulary.add_text_to_vocab("Huge cat")
        self.vocabulary.add_text_to_vocab("Tiny Tiger")
        self.vocabulary.add_text_to_vocab("Huge Tiger")
        self.vocabulary.add_text_to_vocab("tiny cat")

    def test_tokenizing_then_detokenizing_reproduces_the_same_text_when_there_is_no_unknown_word(self):
        text = 'Huge Tiger Tiny cat'
        ids = self.vocabulary.convert_text_to_ids(text)
        self.assertEqual(text.lower(), self.vocabulary.convert_ids_to_text(ids))

    def test_tokenizing_and_detokenizing_replaces_unknown_text_with_unknown_token(self):
        text = 'Huge Tiger and Tiny cat'
        ids = self.vocabulary.convert_text_to_ids(text)
        self.assertEqual("huge tiger <unk> tiny cat", self.vocabulary.convert_ids_to_text(ids))


if __name__ == '__main__':
    unittest.main()
