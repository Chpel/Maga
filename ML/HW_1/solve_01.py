# coding: utf-8
import numpy as np
from typing import SupportsIndex
from sklearn.metrics import classification_report, accuracy_score, precision_score

class KNN(object):
    """
        Класс с реализацией метода ближайших соседей.
    """

    def __init__(self, n_neighbours: int = 4):
        # обучающая выборка: признаки
        self.X_train = None

        # обучающая выборка: метки классов
        self.y_train = None

        # число ближайших соседей
        self.n_neighbours = n_neighbours

    def fit(self, X: np.ndarray, y: SupportsIndex):
        """
            В методе fit (по аналогии с API sklearn) происходит обучение модели.
            Здесь как такового обучения у нас нет, надо просто запомнить датасет
            как "состояние объекта" KNN.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            В методе predict (по аналогии с API sklearn) происходит вычисление предсказанны значений.
        """
        def norms(x0, X):
            """
                Вложенная функция расчёта расстояний от точки x0 (1,2) до массива точек X (n,2)
            """
            return np.linalg.norm(X - x0, axis=1)
        X_norms = np.apply_along_axis(norms, 1, X, self.X_train)
        neighbours = np.argsort(X_norms, axis=1)[:,:self.n_neighbours]
        ys = np.take(self.y_train, neighbours).sum(axis=1)
        pred = ys >= self.n_neighbours/2.
        return pred.astype(int)


def accuracy(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    """
        Доля верно предсказанных меток; это ценная, но далеко не лучшая (помните, почему?)
        оценка качества классификации, но давайте её реализуем
    :param labels_true: одномерный массив int-ов, истинные метки
    :param labels_predicted: одномерный массив int-ов, предсказанные метки
    :return: число совпавших меток делим на общее число меток
    """
    N = labels_true.shape[0]
    trues = (labels_true == labels_predicted).sum()
    return trues/N

def metric(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    """
        Реализуйте какую-нибудь другую метрику качества классификации. Можно
        взять из тех, что были на практике, можно принести какую-то свою.
    :param labels_true: одномерный массив int-ов, истинные метки
    :param labels_predicted: одномерный массив int-ов, предсказанные метки
    :return: число совпавших меток делим на общее число меток
    """
    TP = (labels_true + labels_predicted == 2).sum()
    All_P = (labels_predicted == 1).sum()
    return TP / All_P

def dataplot(X_train, y_train, X_test, y_test, k=0, name=None):
    """
    Функция графика точечных данных из тренировочной и тестовой выборки (или выборки предсказаний)
    :param X_train: тренировочная выборка данных
    :param y_train: тренировочная выборка целевой переменной
    :param X_test: тестовая выборка данных
    :param y_test: тестовая выборка целевой переменной (для отображения истинных классов)
    :param k: число соседей для обучении модели (при k=0 предсказания заменяются истинными классами)
    :param name: имя для сохранения графика данных (если не None)
    """
    plt.figure(figsize=(12,9))
    if k > 0:
        predictor = KNN(n_neighbours=k)
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        plt.title(f"Число соседей: {k}")
    else:
        y_pred = y_test
        plt.title(f"Истинные данные")
    plt.scatter(X_train[:,0], X_train[:,1],c=y_train, marker='*', alpha=0.3, label='train')
    plt.scatter(X_test[:,0], X_test[:,1],c=y_pred, label='pred')
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    if name != None:
        plt.savefig(name + ".png")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # фиксируем random seed для воспроизводимости результата
    np.random.seed(100)

    # создаём синтетический набор данных для обучения и тестирования
    means0 = [1, -1]
    covs0 = [[7, 3],
             [3, 7]]
    x0, y0 = np.random.multivariate_normal(means0, covs0, 190).T

    means1 = [0, -4]
    covs1 = [[0.1, 0.0],
             [0.0, 25]]
    x1, y1 = np.random.multivariate_normal(means1, covs1, 100).T
    data0 = np.vstack([x0, y0]).T
    labels0 = np.zeros(data0.shape[0])

    data1 = np.vstack([x1, y1]).T
    labels1 = np.ones(data1.shape[0])

    data = np.vstack([data0, data1])
    labels = np.hstack([labels0, labels1])
    total_size = data.shape[0]
    print("Original dataset shapes:", data.shape, labels.shape)

    # берём случайные 70% как train
    train_size = int(total_size * 0.7)
    indices = np.random.permutation(total_size)

    # обратите внимание на возможность объявлять несколько переменных в одной строке,
    # бывает удобно, особенно когда переменные связаны по смыслу и когда в правой части короткие выражения
    X_train, y_train = data[indices][:train_size], labels[indices][:train_size]
    X_test, y_test = data[indices][train_size:], labels[indices][train_size:]
    print("Train/test sets shapes:", X_train.shape, X_test.shape)

    # todo: циклом for переберите здесь значения числа ближайших соседей от 1 до 5
    acc_res = []
    prec_res = []
    for k in range(1,6):
        # создаём объект-классификатор
        predictor = KNN(n_neighbours=k)
        print(f"Число соседей: {k}")
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        # Вычислите точность ваших предсказаний.
        print("\tAccuracy: %.4f [ours]" % accuracy(y_test, y_pred))
        assert abs(accuracy_score(y_test, y_pred) - accuracy(y_test, y_pred)) < 1e-5,\
            "Implemented accuracy is not the same as sci-kit learn one!"

        # Проверьте качество вашего классификатора.
        assert accuracy_score(y_test, y_pred) > 19. / 29.,\
            "Your classifier is worse than the constant !"

        # Вычислите какую-то другую метрику (на ваш выбор), сравните с библиотечной версией
        print("\tPrecision: %.4f [ours]" % metric(y_test, y_pred))
        assert abs(metric(y_test, y_pred) - precision_score(y_test, y_pred)) < 1e-5,\
            "Implemented metric is not the same as sci-kit learn one!"

        acc_res.append(accuracy(y_test, y_pred))
        prec_res.append(metric(y_test, y_pred))
    #   и подготовьте ТРИ картинки по ТЕСТОВОЙ ВЫБОРКЕ: исходные метки, метки с n_neighbours = 1
    #   и лучшим n_neighbours в рассмотренном нами интервале. Один класс -- одним цветом, другой 
    dataplot(X_train, y_train, X_test, y_test, k=0, name="true")
    dataplot(X_train, y_train, X_test, y_test, k=1, name="kNN_1")
    dataplot(X_train, y_train, X_test, y_test, k=5, name="kNN_5_best")
    #ЧЕТВЕРТАЯ кактинка -- график зависимости метрик от числа соседей.
    plt.figure(figsize=(8,6))
    x = range(1,6)
    plt.plot(x, acc_res, label='accuracy')
    plt.plot(x, prec_res, label='metric (precision)')
    plt.xlabel('Число соседей')
    plt.ylabel('Метрика')
    plt.grid()
    plt.xticks(x)
    plt.legend();
    plt.savefig("metrics.png")

    
    
