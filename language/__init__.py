from language.abc import AbstractBaseVocabulary
from language.tokenizers import BasicTokenizer
from language.utils import create_read_func
from language.vocabulary import SimpleVocabulary


# TODO: Automatically generate vocab file
def vocabulary_factory(config):
    vocab_path = config['vocab_path']
    vocab_threshold = config['vocab_threshold']

    read_func = create_read_func(vocab_path)

    vocab = SimpleVocabulary.create_vocabulary_from_storage(read_func)
    vocab.threshold_rare_words(vocab_threshold)
    return vocab
