import MetalNeedle as mn


class MnistDataLoader:
    """ Load mnist data set """
    def __init__(self, images_file, label_file, batch_size, shuffle=True):
        # meta data
        self.images_file = images_file
        self.label_file = label_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_header_size = 16
        self.label_header_size = 8
        self.num_images, self.num_rows, self.num_cols = self._read_image_metadata(images_file)
        self.num_labels = self._read_label_metadata(label_file)
        if self.num_images != self.num_labels:
            raise ValueError(
                f"mnist image count {self.num_images} != label count {self.num_labels}"
            )

        # keep track of where we are in data
        self.idx = 0

    def _read_image_metadata(self, file):
        with open(file, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            return num_images, num_rows, num_cols

    def _read_label_metadata(self, file):
        with open(file, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            return num_labels

    def _read_batch(self, start: int, end: int):
        current_batch_size = end - start
        image_size = self.num_rows * self.num_cols
        with open(self.images_file, 'rb') as image_file, open(self.label_file, 'rb') as label_file:
            image_file.seek(self.image_header_size + start * image_size)
            label_file.seek(self.label_header_size + start)

            image_buffer = image_file.read(current_batch_size * image_size)
            label_buffer = label_file.read(current_batch_size)

        images = mn.Tensor.load_from_buffer(
            image_buffer,
            [current_batch_size, image_size],
            dtype="float32",
            requires_grad=False,
            normalize=255.0,
        )
        labels = mn.Tensor.load_from_buffer(
            label_buffer,
            [current_batch_size],
            dtype="float32",
            normalize=1.0,
        ).one_hot(10)
        return images, labels

    def num_batches(self) -> int:
        return self.num_images // self.batch_size

    def reset(self) -> None:
        self.idx = 0

    def iter_batches(self, max_batches=None):
        self.reset()
        total_batches = self.num_batches()
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        for _ in range(total_batches):
            start = self.idx
            end = start + self.batch_size
            self.idx = end
            yield self._read_batch(start, end)
