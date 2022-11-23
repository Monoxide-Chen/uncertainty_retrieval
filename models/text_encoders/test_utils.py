import unittest

import torch

from models.text_encoders.utils import retrieve_last_timestamp_output


class TestTextModelUtils(unittest.TestCase):
    def test_retrieve_last_timestamp_output(self):
        batch_size, max_seq_len, feature_size = 4, 5, 3
        lstm_outputs = torch.rand([batch_size, max_seq_len, feature_size])
        last_timestamps = torch.LongTensor([3, 1, 4, 2])

        outputs = retrieve_last_timestamp_output(lstm_outputs, last_timestamps)

        for output, lstm_output, last_timestamp in zip(outputs, lstm_outputs, last_timestamps):
            self.assertTrue(output.equal(lstm_output[last_timestamp - 1].squeeze(0)))


if __name__ == '__main__':
    unittest.main()
