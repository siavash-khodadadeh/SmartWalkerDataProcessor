import numpy as np


def window_stack(data, step_size=1, width=3):
    data, labels = data[:, :4], data[:, 4]
    assert(width % 2 != 0)
    n = data.shape[0]
    result = np.hstack(data[i:1 + n + i - width:step_size] for i in range(0, width))
    result = result.reshape(n - width + 1, width, 4)
    return result, labels[int(width/2):int(-width/2)].reshape(-1, 1)


def balance_data(data, labels):
    one_labels = data[np.where(labels == 1)[0], :, :]
    n = one_labels.shape[0]
    zero_labels_indices = np.where(labels == 0)[0]
    np.random.shuffle(zero_labels_indices)
    zero_labels_indices = zero_labels_indices[:n]
    zero_labels = data[zero_labels_indices, :, :]
    balanced_data = np.concatenate((zero_labels, one_labels))
    balanced_label = np.zeros(shape=(balanced_data.shape[0], 1))
    balanced_label[n:] = 1
    return balanced_data, balanced_label


def get_data_and_labels(data_address, preprocessed_file_address, step_size=1, width=5, train_ratio=0.8):
    preprocessed_file = open(preprocessed_file_address, 'w')
    with open(data_address) as original_file:
        for line in original_file:
            new_line = line.replace(',', '').replace('False', '0').replace('True', '1').replace('\n', '')
            preprocessed_file.write(new_line + '\n')

    preprocessed_data = np.loadtxt(preprocessed_file_address)
    data, labels = window_stack(preprocessed_data, step_size=step_size, width=width)

    data[data > 1000] = 1000
    data[data < -1000] = -1000
    data /= 1000.

    pivot = int(train_ratio * data.shape[0])
    train_data, train_labels = data[:pivot,: ,:], labels[:pivot, :]
    validation_data, validation_labels = data[pivot:, :, :], labels[pivot:, :]

    train_data, train_labels = balance_data(train_data, train_labels)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    train_data = train_data[indices, :, :]
    train_labels = train_labels[indices, :]

    return train_data, train_labels, validation_data, validation_labels
