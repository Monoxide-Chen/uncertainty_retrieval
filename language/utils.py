import pickle


def create_read_func(vocab_path):
    def read_func():
        with open(vocab_path, 'rb') as f:
            data = pickle.load(f)
        return data

    return read_func


def create_write_func(vocab_path):
    def write_func(data):
        with open(vocab_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return write_func
