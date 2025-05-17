import MetalNeedle as mn


class MnistDataLoader:
    """ Load mnist data set """
    def __init__(self, images_file, label_file, batch_size, shuffle=True):
        # meta data
        self.batch_size = batch_size
        self.shuffle = shuffle

        # load different data
        self.images = self._read_images(images_file)
        self.labels = self._read_labels(label_file)

        # keep track of where we are in data
        self.idx = 0

    def _read_images(self, file):
        with open(file, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')

            buffer = f.read(self.batch_size * num_rows * num_cols)

            tensor = mn.Tensor.load_from_buffer(buffer, [self.batch_size, num_cols * num_rows], dtype="float32", requires_grad=True)
            normalized_tensor = tensor / 255
            return normalized_tensor

    def _read_labels(self, file):
        with open(file, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')

            buffer = f.read(self.batch_size)
            labels =  mn.Tensor.load_from_buffer(buffer, [self.batch_size], dtype="float32")
            return labels.one_hot(10)