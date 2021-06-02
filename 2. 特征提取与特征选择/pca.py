from sklearn.datasets.base import load_iris
import sklearn.decomposition as dp
import matplotlib.pyplot as plt
import numpy as np


def show_pca():
    x, y = load_iris(return_X_y=True)
    print(x.shape)
    print(y.shape)
    pca = dp.PCA(n_components=2)
    reduced_x = pca.fit_transform(x)
    print(reduced_x.shape)
    print(y)
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])

    plt.scatter(red_x, red_y, color='r', marker='.')
    plt.scatter(blue_x, blue_y, color='b', marker='D')
    plt.scatter(green_x, green_y, color='g', marker='x')
    plt.show()


class PCAx():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.features = X.shape[0]
        # 1. 协方差矩阵
        # 1.1 去中心化
        X_0 = X-X.mean(axis=0)
        # print(X_0.shape)
        # 1.2 协方差矩阵
        # pprint.pprint(X_0.T.dot(X_0))
        self.convariance = X_0.T.dot(X_0)/X_0.shape[0]
        #  对应特征值及特征向量
        eig_vals, eig_vector = np.linalg.eig(self.convariance)
        # print(eig_vals, eig_vector)

        # 特征值排序
        idx = np.argsort(-eig_vals)
        print(idx)
        # 降维矩阵
        self.n_components_ = eig_vector[:, idx[:self.n_components]]
        # 矩阵降维
        return X_0.dot(self.n_components_)


def pca_sklearn(X):
    pca = dp.PCA(n_components=2)
    x_pca = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)  # 输出贡献率
    return x_pca


if __name__ == '__main__':
    # show_pca()

    array_list = [
        [-1, 2, 66, -1],
        [-2, 6, 58, -1],
        [-3, 8, 45, -2],
        [1, 9, 36, 1],
        [2, 10, 62, 1],
        [3, 5, 83, 2],
    ]
    import pprint

    X = np.array(array_list)  # 导入数据，维度为4

    pca = PCAx(n_components=2)
    pca_x = pca.fit_transform(X)
    pprint.pprint(pca_x)

    newX = pca_sklearn(X)
    pprint.pprint(newX)
