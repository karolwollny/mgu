import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import model_from_json

TRAIN_DIR = './../data/asl_alphabet_train/asl_alphabet_train'
TEST_DIR = './../data/asl_alphabet_test/asl_alphabet_test'

LABELS_DICT = {'A': 0,  'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,    'L': 11,
               'M': 12, 'N': 13,    'O': 14,    'P': 15,    'Q': 16,    'R': 17,    'S': 18,    'T': 19,    'U': 20,
               'V': 21, 'W': 22,    'X': 23,    'Y': 24,    'Z': 25,    'del': 26,  'nothing': 27,  'space': 28}


def load_data(data_folder_path):
    x, y = [], []
    for folder_name in os.listdir(data_folder_path):
        if folder_name in LABELS_DICT:
            label = LABELS_DICT[folder_name]
        else:
            label = 29
        for image_filename in os.listdir(os.path.join(data_folder_path, folder_name)):
            image = cv2.imread(os.path.join(data_folder_path, folder_name, image_filename))
            if image is not None:
                image_array = np.asarray(image)
                x.append(image_array)
                y.append(label)
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")


def load_model(filename='model.json', h5_filename='model.h5'):
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read();
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_filename)
    return loaded_model


if __name__ == "__main__":
    x_train, y_train = load_data(TRAIN_DIR)

    # plt.figure(figsize=(1, 1))
    # plt.imshow(X_train[100])
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    # Zamienienie macierza zdjęcia na wektor
    x_train = x_train.reshape(-1, 32*32*3)
    x_test = x_test.reshape(-1, 32*32*3)

    # print(x_train.shape)
    # print(x_test.shape)

    # Normalizacja
    x_train.astype(np.float32)
    x_test.astype(np.float32)
    x_train = x_train/255.0
    x_test = x_test/255.0

    # One-hot (label=3 -> label=[0. 0. 0. 1. 0. 0. <...> 0.])
    y_train = keras.utils.to_categorical(y_train, 30)
    y_test = keras.utils.to_categorical(y_test, 30)

    # print(y_train[10])

    model = keras.models.Sequential()
    model.add(Dense(512, input_shape=(32*32*3, )))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(30))
    model.add(Activation('softmax'))

    # model.summary()
    # plot_model(model)
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')

