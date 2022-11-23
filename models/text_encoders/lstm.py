import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.abc import AbstractBaseTextEncoder
from models.text_encoders.utils import retrieve_last_timestamp_output


class SimpleLSTMEncoder(AbstractBaseTextEncoder):
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__(vocabulary_len, padding_idx, feature_size, *args, **kwargs)
        word_embedding_size = kwargs.get('word_embedding_size', 512)
        lstm_hidden_size = kwargs.get('lstm_hidden_size', 512)
        feature_size = feature_size

        self.embedding_layer = nn.Embedding(vocabulary_len, word_embedding_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(word_embedding_size, lstm_hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(lstm_hidden_size, feature_size),
        )

    def forward(self, x, lengths):
        # x is a tensor that has shape of (batch_size * seq_len)
        x = self.embedding_layer(x)  # x's shape (batch_size * seq_len * word_embed_dim)
        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(x)
        outputs = retrieve_last_timestamp_output(lstm_outputs, lengths)

        outputs = self.fc(outputs)
        return outputs

    @classmethod
    def code(cls) -> str:
        return 'lstm'


class NormalizationLSTMEncoder(SimpleLSTMEncoder):
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__(vocabulary_len, padding_idx, feature_size, *args, **kwargs)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, x: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        outputs = super().forward(x, lengths)
        return F.normalize(outputs) * self.norm_scale

    @classmethod
    def code(cls) -> str:
        return 'norm_lstm'


class SimplerLSTMEncoder(AbstractBaseTextEncoder):
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__(vocabulary_len, padding_idx, feature_size, *args, **kwargs)
        word_embedding_size = kwargs.get('word_embedding_size', 512)

        self.embedding_layer = nn.Embedding(vocabulary_len, word_embedding_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(word_embedding_size, self.feature_size, batch_first=True)

    def forward(self, x, lengths):
        # x is a tensor that has shape of (batch_size * seq_len)
        x = self.embedding_layer(x)  # x's shape (batch_size * seq_len * word_embed_dim)
        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(x)
        return retrieve_last_timestamp_output(lstm_outputs, lengths)

    @classmethod
    def code(cls) -> str:
        return 'simpler_lstm'
