# coding: utf-8

from sklearn.datasets import make_moons


def get_moon_data():
    X, y = make_moons(n_samples=200,
                      noise=0.05,
                      random_state=0)
    return (X, y)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X, y = get_moon_data()
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
