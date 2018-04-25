import numpy as np


class Dataset(object):
    def __init__(self, data, labels, shuffle=True, default_batch_size=64):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.data = data
        self.labels = labels
        self._num_examples = len(data)
        self.shuffle = shuffle
        self.default_batch_size = default_batch_size

    def __len__(self):
        return self._num_examples

    def get_epochs_completed(self):
        return self._epochs_completed

    def reset_and_shuffle(self):
        idx = np.arange(0, self._num_examples)
        if self.shuffle:
            np.random.shuffle(idx)

        self.data = self.data[idx, :]
        self.labels = self.labels[idx, :]

    def divide(self, per_class_sample=15):
        idx = list()
        for label in range(100):
            label_idx = np.where(self.labels == label)[0]
            np.random.shuffle(label_idx)
            label_idx = label_idx[:per_class_sample]
            idx.extend(label_idx)

        self.data = self.data[idx, :, :, :]
        self.labels = self.labels[idx]
        self._num_examples = len(idx)
        self.reset_and_shuffle()

    def cross_validation(self, per_class_sample=450):
        idx = list()
        for label in range(100):
            label_idx = np.where(self.labels == label)[0]
            np.random.shuffle(label_idx)
            label_idx = label_idx[:per_class_sample]
            idx.extend(label_idx)

        images = self.data[idx, :, :, :]
        labels = self.labels[idx]

        not_idx = np.setxor1d(idx, np.arange(0, self._num_examples))
        val_images = self.data[not_idx, :, :, :]
        val_labels = self.labels[not_idx]

        train_dataset = Dataset(images, labels, shuffle=self.shuffle, default_batch_size=self.default_batch_size)
        validation_dataset = Dataset(
            val_images,
            val_labels,
            shuffle=self.shuffle,
            default_batch_size=self.default_batch_size
        )

        return train_dataset, validation_dataset

    def next_batch(self, batch_size=None):
        if not batch_size:
            batch_size = self.default_batch_size

        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            self.reset_and_shuffle()

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self.data[start:self._num_examples, :]
            labels_rest_part = self.labels[start:self._num_examples, :]

            self.reset_and_shuffle()

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.data[start:end, :]
            labels_new_part = self.labels[start:end, :]

            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.data[start:end, :], self.labels[start:end, :]
