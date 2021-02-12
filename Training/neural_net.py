import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# input 2 units (x, y)
# hidden 1: 50 units
# hidden 2: 30 units
# FC1, FC2, FC3


def create_model():
    inputs = tf.keras.Input(shape=(2))
    x = tf.keras.layers.Dense(2000, activation='relu', name='h1')(inputs)
    x = tf.keras.layers.Dense(1200, activation='relu', name='h2')(x)
    x = tf.keras.layers.Dense(600, activation='relu', name='h3')(x)
    a = tf.keras.layers.Dense(2, activation='softmax', name='a')(x)
    b = tf.keras.layers.Dense(2, activation='softmax', name='b')(x)
    c = tf.keras.layers.Dense(2, activation='softmax', name='c')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[a, b, c], name='HierClassifier2D')
    return model


def import_dataset(path):
    df = pd.read_csv(path)
    target = []
    points = []
    for index, row in df.iterrows():
        element = np.array([row['x'], row['y']])
        points.append(element)
        target.append(int(row['label'])-1)

    return np.array(points), target

def create_targets(target):
    target_a = []
    target_b = []
    target_c = []

    for i in range(len(target)):
        if target[i] == 0:
            target_a.append(0)
            target_b.append(0)
            target_c.append(-1)
        elif target[i] == 1:
            target_a.append(0)
            target_b.append(1)
            target_c.append(-1)
        elif target[i] == 2:
            target_a.append(1)
            target_b.append(-1)
            target_c.append(0)
        elif target[i] == 3:
            target_a.append(1)
            target_b.append(-1)
            target_c.append(1)
        else:
            print(target[i])
            print("oops...")
            exit(-1)
    return target_a, target_b, target_c

def to_categorical_custom(array, num_classes=2):
    categorical = []
    for i in range(len(array)):
        if array[i] == -1:
            categorical.append(np.zeros(num_classes, dtype=np.float))
        elif array[i] == 0:
            a = np.zeros(num_classes, dtype=np.float)
            a[0] = float(1)
            categorical.append(a)
        elif array[i] == 1:
            a = np.zeros(num_classes, dtype=np.float)
            a[1] = float(1)
            categorical.append(a)
        elif array[i] == 2:
            a = np.zeros(num_classes, dtype=np.float)
            a[2] = float(1)
            categorical.append(a)
    return categorical

def to_cat(a, b, c):
    a = to_categorical_custom(a, 2)
    b = to_categorical_custom(b, 2)
    c = to_categorical_custom(c, 2)
    return a, b, c

def plot_val_train_error(fit):
    plt.plot(fit.history['a_accuracy'])
    plt.plot(fit.history['b_accuracy'])
    plt.plot(fit.history['c_accuracy'])
    plt.plot(fit.history['val_a_accuracy'])
    plt.plot(fit.history['val_b_accuracy'])
    plt.plot(fit.history['val_c_accuracy'])
    plt.grid(True)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(fit.history['loss'])
    plt.plot(fit.history['a_loss'])
    plt.plot(fit.history['b_loss'])
    plt.plot(fit.history['c_loss'])
    plt.plot(fit.history['val_a_loss'])
    plt.plot(fit.history['val_b_loss'])
    plt.plot(fit.history['val_c_loss'])
    plt.grid(True)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':

    start_time = time.time()
    model = create_model()
    x_train, y_train = import_dataset('/home/ruben/PycharmProjects/HierClass2D/Dataset/dataset2/train.csv')
    print(y_train)
    x_val, y_val = import_dataset('/home/ruben/PycharmProjects/HierClass2D/Dataset/dataset2/val.csv')
    x_test, y_test = import_dataset('/home/ruben/PycharmProjects/HierClass2D/Dataset/dataset2/test.csv')

    y_a_train, y_b_train, y_c_train = create_targets(y_train)
    print("hey")
    y_a_train_cat, y_b_train_cat, y_c_train_cat = to_cat(y_a_train, y_b_train, y_c_train)

    y_a_val, y_b_val, y_c_val = create_targets(y_val)
    print("hei")
    y_a_val_cat, y_b_val_cat, y_c_val_cat = to_cat(y_a_val, y_b_val, y_c_val)

    losses = {
        "a": "categorical_crossentropy",
        "b": "categorical_crossentropy",
        "c": "categorical_crossentropy"
    }

    lr = 1e-6
    batch_size = 20
    no_epochs = 1000

    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss=losses, metrics=['accuracy'], run_eagerly=True)

    checkpoint_path = "/home/ruben/Desktop/Hier2D_tese/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, save_best_only=True)
    
    fit = model.fit(x_train, [y_a_train_cat, y_b_train_cat, y_c_train_cat], batch_size=batch_size, epochs=no_epochs, callbacks=[cp_callback], shuffle=True, validation_data=(x_val, [y_a_val_cat, y_b_val_cat, y_c_val_cat])) # class_weights?

    plot_val_train_error(fit)

    model.load_weights(checkpoint_path)

    print("--- %s seconds ---" % (time.time() - start_time))
