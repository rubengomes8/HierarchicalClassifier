import pandas as pd
from Gaussian import Gaussian
import random

def params(range_mu_x, range_mu_y, cov_11, cov_12, cov_21, cov_22):
    mu_x = random.uniform(range_mu_x[0], range_mu_x[1])
    mu_y = random.uniform(range_mu_y[0], range_mu_y[1])
    mean = [mu_x, mu_y]

    cov = [[cov_11, cov_12], [cov_21, cov_22]]
    return mean, cov

def create_dataset(no_examples, g1, g2, g3, g4):
    list_x = []
    list_y = []
    list_label = []
    for i in range(0, int(no_examples/4)):
        (x, y) = g1.gauss_2d()
        list_x.append(x)
        list_y.append(y)
        list_label.append(1)
        (x, y) = g2.gauss_2d()
        list_x.append(x)
        list_y.append(y)
        list_label.append(2)
        (x, y) = g3.gauss_2d()
        list_x.append(x)
        list_y.append(y)
        list_label.append(3)
        (x, y) = g4.gauss_2d()
        list_x.append(x)
        list_y.append(y)
        list_label.append(4)

    # shuffle patterns in the same way
    c = list(zip(list_x, list_y, list_label))
    random.shuffle(c)
    list_x, list_y, list_label = zip(*c)

    return list(list_x), list(list_y), list(list_label)

def write_csv(path, list_x, list_y, list_label):
    d = {'x': list_x, 'y': list_y, 'label': list_label}
    df = pd.DataFrame(d)
    # print(df)
    df.to_csv(path)

if __name__ == '__main__':

    mean_1, cov_1 = params((-7, -5), (4, 6), 0.5, 2, 2, 0.8)
    print(mean_1, cov_1)
    g1 = Gaussian(mean_1, cov_1)

    mean_2, cov_2 = params((0, 2), (3, 6), 2, 4, 4, 2)
    print(mean_2, cov_2)
    g2 = Gaussian(mean_2, cov_2)

    mean_3, cov_3 = params((4, 7), (-5, -3), 1, 3, 3, 1)
    print(mean_3, cov_3)
    g3 = Gaussian(mean_3, cov_3)

    mean_4, cov_4 = params((-5, -2), (-6, -3), 0, 4, 4, 0)
    print(mean_4, cov_4)
    g4 = Gaussian(mean_4, cov_4)

    list_x, list_y, list_label = create_dataset(2000, g1, g2, g3, g4) # train
    write_csv('/home/ruben/PycharmProjects/HierClassifier2D/Dataset/dataset2/train.csv', list_x, list_y, list_label)

    list_x, list_y, list_label = create_dataset(300, g1, g2, g3, g4) # val
    write_csv('/home/ruben/PycharmProjects/HierClassifier2D/Dataset/dataset2/val.csv', list_x, list_y, list_label)

    list_x, list_y, list_label = create_dataset(600, g1, g2, g3, g4) # test
    write_csv('/home/ruben/PycharmProjects/HierClassifier2D/Dataset/dataset2/test.csv', list_x, list_y, list_label)


