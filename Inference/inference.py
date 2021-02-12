import tensorflow as tf
import pandas as pd
import numpy as np

def import_dataset(path):
    df = pd.read_csv(path)
    target = []
    points = []
    for index, row in df.iterrows():
        element = np.array([row['x'], row['y']])
        points.append(element)
        target.append(int(row['label']))

    return np.array(points), target

def create_model():
    inputs = tf.keras.Input(shape=(2))
    x = tf.keras.layers.Dense(50, activation='relu', name='h1')(inputs)
    x = tf.keras.layers.Dense(30, activation='relu', name='h2')(x)
    a = tf.keras.layers.Dense(2, activation='softmax', name='a')(x)
    b = tf.keras.layers.Dense(2, activation='softmax', name='b')(x)
    c = tf.keras.layers.Dense(2, activation='softmax', name='c')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[a, b, c], name='HierClassifier2D')
    return model

def create_pred(y):
    y_pred_a = []
    y_pred_b = []
    y_pred_c = []
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j][0] > y[i][j][1]:
                if i == 0:
                    y_pred_a.append(0)
                elif i == 1:
                    y_pred_b.append(0)
                elif i == 2:
                    y_pred_c.append(0)
                else:
                    print(i)
                    print("oops!")
                    exit(-1)
            else:
                if i == 0:
                    y_pred_a.append(1)
                elif i == 1:
                    y_pred_b.append(1)
                elif i == 2:
                    y_pred_c.append(1)
                else:
                    print(i)
                    print("oops")
                    exit(-1)
    return y_pred_a, y_pred_b, y_pred_c

def accuracy_a(pred, true):
    total = len(pred)
    sucessos = 0
    for i in range(len(pred)):
        if true[i] == 1 or true[i] == 2:
            if pred[i] == 0:
                sucessos += 1
        else:
            if pred[i] == 1:
                sucessos += 1
    return sucessos/total

def accuracy_b(pred, true):
    total = len(pred)
    sucessos = 0
    for i in range(len(pred)):
        if true[i] == 1:
            if pred[i] == 0:
                sucessos += 1
        elif true[i] == 2:
            if pred[i] == 1:
                sucessos += 1
    return sucessos/total

def accuracy_c(pred, true):
    total = len(pred)
    sucessos = 0
    for i in range(len(pred)):
        if true[i] == 3:
            if pred[i] == 0:
                sucessos += 1
        elif true[i] == 4:
            if pred[i] == 1:
                sucessos += 1
    return sucessos/total

def filter_pred(true, pred, a, b):
    new_true = []
    new_pred = []
    for i in range(len(true)):
        if true[i] == a or true[i] == b:
            new_true.append(true[i])
            new_pred.append(pred[i])

    return new_true, new_pred

if __name__ == '__main__':

    x_val, y_val = import_dataset('/home/ruben/PycharmProjects/HierClassifier2D/Dataset/dataset1/val.csv')
    model = create_model()
    checkpoint_path = "/home/ruben/Desktop/Hier2D/cp.ckpt"
    model.load_weights(checkpoint_path)
    y_pred = model.predict(x_val)
    print(y_pred)
    y_pred_a, y_pred_b, y_pred_c = create_pred(y_pred)

    y_true_b, y_pred_b = filter_pred(y_val, y_pred_b, 1, 2)
    y_true_c, y_pred_c = filter_pred(y_val, y_pred_c, 3, 4)
    print(accuracy_a(y_pred_a, y_val))
    print(accuracy_b(y_pred_b, y_true_b))
    print(accuracy_c(y_pred_c, y_true_c))
