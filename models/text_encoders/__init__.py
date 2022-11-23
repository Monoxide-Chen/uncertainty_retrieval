from language.vocabulary import AbstractBaseVocabulary
from trainers.abc import AbstractBaseTextEncoder
from models.text_encoders.lstm import SimpleLSTMEncoder, NormalizationLSTMEncoder, SimplerLSTMEncoder
from models.text_encoders.roberta import RobertaEncoder, BertFc

TEXT_MODEL_CODES = [SimpleLSTMEncoder.code(), NormalizationLSTMEncoder.code(), SimplerLSTMEncoder.code()]


def text_encoder_factory(vocabulary: AbstractBaseVocabulary, config: dict) -> AbstractBaseTextEncoder:
    model_code = config['text_encoder']
    feature_size = config['text_feature_size']
    word_embedding_size = config['word_embedding_size']
    lstm_hidden_size = 512

    if model_code == SimpleLSTMEncoder.code():
        return SimpleLSTMEncoder(vocabulary_len=len(vocabulary), padding_idx=vocabulary.pad_id(),
                                 feature_size=feature_size, word_embedding_size=word_embedding_size,
                                 lstm_hidden_size=lstm_hidden_size), None
    elif model_code == NormalizationLSTMEncoder.code():
        return NormalizationLSTMEncoder(vocabulary_len=len(vocabulary), padding_idx=vocabulary.pad_id(),
                                        feature_size=feature_size, norm_scale=config['norm_scale'],
                                        word_embedding_size=word_embedding_size,
                                        lstm_hidden_size=lstm_hidden_size), None
    elif model_code == SimplerLSTMEncoder.code():
        return SimplerLSTMEncoder(vocabulary_len=len(vocabulary), padding_idx=vocabulary.pad_id(),
                                  feature_size=feature_size, word_embedding_size=word_embedding_size,
                                  lstm_hidden_size=lstm_hidden_size), None
    elif model_code == RobertaEncoder.code():
        return RobertaEncoder(feature_size=feature_size), BertFc(feature_size=feature_size)
    else:
        raise ValueError("There's no text encoder matched with {}".format(model_code))
