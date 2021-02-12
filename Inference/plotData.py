import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_csv(p_train, p_val, p_test):
    df_train = pd.read_csv(p_train)
    # df_val = pd.read_csv(p_val)
    df_val = None
    # df_test = pd.read_csv(p_test)
    df_test = None
    return df_train, df_val, df_test

def create_dataset_for_df(df, d1, d2, d3, d4):
    if df is not None:
        for index, row in df.iterrows():
            if int(row['label']) == 1:
                d1.append((row['x'], row['y'], int(row['label'])))
            elif int(row['label']) == 2:
                d2.append((row['x'], row['y'], int(row['label'])))
            elif int(row['label']) == 3:
                d3.append((row['x'], row['y'], int(row['label'])))
            elif int(row['label']) == 4:
                d4.append((row['x'], row['y'], int(row['label'])))
            else:
                print("Oops")
                exit(-1)
        return d1, d2, d3, d4
    else:
        return -1

def create_dataset(df_train, df_val, df_test):
    d1 = []
    d2 = []
    d3 = []
    d4 = []

    d1, d2, d3, d4 = create_dataset_for_df(df_train, d1, d2, d3, d4)
    if df_val is not None:
        d1, d2, d3, d4 = create_dataset_for_df(df_val, d1, d2, d3, d4)
    if df_test is not None:
        d1, d2, d3, d4 = create_dataset_for_df(df_test, d1, d2, d3, d4)

    return d1, d2, d3, d4

def create_lists_from_tuples(list_tuples):
    list_x = []
    list_y = []
    list_label = []

    for i in range(len(list_tuples)):
        list_x.append(list_tuples[i][0])
        list_y.append(list_tuples[i][1])
        list_label.append(list_tuples[i][2])

    return list_x, list_y, list_label


if __name__ == '__main__':

    p_train = '/home/ruben/PycharmProjects/HierClass2D/Dataset/dataset2/train.csv'
    p_val = '/home/ruben/PycharmProjects/HierClass2D/Dataset/dataset2/val.csv'
    p_test = '/home/ruben/PycharmProjects/HierClass2D/Dataset/dataset2/test.csv'

    df_train, df_val, df_test = read_csv(p_train, p_val, p_test)
    d1, d2, d3, d4 = create_dataset(df_train, df_val, df_test)

    list_x, list_y, _ = create_lists_from_tuples(d1)
    plt.plot(list_x, list_y, 'co')

    list_x, list_y, _ = create_lists_from_tuples(d2)
    plt.plot(list_x, list_y, 'bo')

    list_x, list_y, _ = create_lists_from_tuples(d3)
    plt.plot(list_x, list_y, 'go')

    list_x, list_y, _ = create_lists_from_tuples(d4)
    plt.plot(list_x, list_y, 'ko')
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)

    plt.xticks([-10, -5, 0, 5, 10], fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
