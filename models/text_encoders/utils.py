import torch


def retrieve_last_timestamp_output(lstm_outputs: torch.Tensor, lengths: torch.LongTensor, timestamp_dim=1):
    batch_size, max_seq_len, lstm_hidden_dim = lstm_outputs.size()

    last_timestamps = (lengths - 1).view(-1, 1).expand(batch_size, lstm_hidden_dim)  # (batch_size, feature_size)
    last_timestamps = last_timestamps.unsqueeze(timestamp_dim)  # (batch_size, 1, feature_size)
    return lstm_outputs.gather(timestamp_dim, last_timestamps).squeeze(1)  # (batch_size, feature_size)
