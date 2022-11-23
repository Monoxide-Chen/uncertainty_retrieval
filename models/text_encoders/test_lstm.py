import unittest

import torch

from models.text_encoders.lstm import SimpleLSTMEncoder, NormalizationLSTMEncoder, SimplerLSTMEncoder


class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.vocabulary_size = 10
        self.batch_size, self.max_seq_len, self.feature_size = 4, 5, 3
        self.word_embedding_size, self.lstm_hidden_size, self.fc_output_size = 5, 10, 11
        self.norm_scale = 4
        self.epsilon = 1e-4

        self.some_big_scala = 10000
        self.inputs = torch.randint(0, self.vocabulary_size, [self.batch_size, self.max_seq_len])

    def test_the_input_output_size_of_simple_lstm_model(self):
        model = SimpleLSTMEncoder(self.vocabulary_size, padding_idx=0, word_embedding_size=self.word_embedding_size,
                                  lstm_hidden_size=self.lstm_hidden_size, feature_size=self.fc_output_size)
        lengths = torch.LongTensor([3, 1, 2, 4])

        outputs = model(self.inputs, lengths)
        self.assertTupleEqual((self.batch_size, self.fc_output_size), outputs.size())

    def test_the_input_output_size_of_simpler_lstm_model(self):
        model = SimplerLSTMEncoder(self.vocabulary_size, padding_idx=0, word_embedding_size=self.word_embedding_size,
                                   lstm_hidden_size=self.lstm_hidden_size, feature_size=self.fc_output_size)
        lengths = torch.LongTensor([3, 1, 2, 4])

        outputs = model(self.inputs, lengths)
        self.assertTupleEqual((self.batch_size, self.fc_output_size), outputs.size())

    def test_the_input_output_size_of_norm_lstm_model(self):
        model = NormalizationLSTMEncoder(self.vocabulary_size, padding_idx=0,
                                         word_embedding_size=self.word_embedding_size,
                                         lstm_hidden_size=self.lstm_hidden_size,
                                         feature_size=self.fc_output_size, norm_scale=self.norm_scale)
        lengths = torch.LongTensor([3, 1, 2, 4])

        outputs = model(self.inputs, lengths)

        self.assertTupleEqual((self.batch_size, self.fc_output_size), outputs.size())
        self.assertEqual(self.batch_size, torch.sum(torch.norm(outputs, dim=1) < self.norm_scale + self.epsilon))

    def test_the_input_output_size_of_batch_size1_lstm_model2(self):
        batch_size, max_seq_len, feature_size = 1, 14, 3
        inputs = torch.randint(0, self.vocabulary_size, [batch_size, max_seq_len])

        model = SimpleLSTMEncoder(self.vocabulary_size, padding_idx=0,
                                   word_embedding_size=self.word_embedding_size,
                                   lstm_hidden_size=self.lstm_hidden_size,
                                   feature_size=self.fc_output_size, norm_scale=self.norm_scale)
        lengths = torch.LongTensor([14])

        outputs = model(inputs, lengths)

        self.assertTupleEqual((batch_size, self.fc_output_size), outputs.size())
        self.assertEqual(batch_size, torch.sum(torch.norm(outputs, dim=1) < self.norm_scale + self.epsilon))


if __name__ == '__main__':
    unittest.main()
