"""
Multiplication rule
p(x,y) = p(y|x) * p(x)

p(y|x) = p(x,y)
         ------
          p(x)

p(x,y) = p(x|y) * p(y)

Bayes rule
p(y|x) = p(x|y) * p(y)
         -------------
              p(x)

Independence assumption -> naive bayes
p(y_j|x) = \prod_i{p(x_i|y_j)} * p(y_j)

\log{p(y_j|x)} = \log{\prod_i{p(x_i|y_j)} * p(y_j)}

\log{p(y_j|x)} = \sum_i{\log{p(x_i|y_j)}} + \log{p(y_j)}

Log likelihood -> linear classifier
\log{p(y_j|x)} = (x @ w_j) + b_j

https://en.wikipedia.org/wiki/Naive_Bayes_classifier
"""


import numpy as np
np.random.seed(123)
# from sklearn.datasets import load_iris as load_data_func
from sklearn.datasets import load_wine as load_data_func
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler


def load_classification_data():
    """Load classification data.

    Returns:
        {train,val}_x : nsamples x nfeatures
        {train,val}_y : nsamples
    """
    data = load_data_func(as_frame=True)

    print('class frequencies', data.target.value_counts())

    x = np.asarray(data.data)
    y = np.asarray(data.target)

    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    split_index = int(len(x)*0.8)
    train_indexes, val_indexes = indexes[:split_index], indexes[split_index:]

    train_x = x[train_indexes]
    train_y = y[train_indexes]

    val_x = x[val_indexes]
    val_y = y[val_indexes]

    return train_x, train_y, val_x, val_y


def gaussian_pdf(x, mean, std):
    return np.exp(-1 * ((x-mean)**2) / (2*(std**2)) ) / np.sqrt(2*np.pi*(std**2))


def stack_posterior(posterior_by_classid, classids):
    return np.stack(
        [posterior_by_classid[classid] for classid in classids],
        axis=1
    )


class GaussianNB(object):
    def __init__(self):
        self.fitted = False

    def fit(self, train_x, train_y):
        """Fit the model.

        Arguments:
            train_x : nsamples x nfeatures
            train_y : nsamples
        """
        if self.fitted:
            raise
        self.fitted = True

        self.classids = sorted(set(train_y))
        self.mean_by_classid = {}
        self.std_by_classid = {}
        self.class_freqs = {}
        for classid in self.classids:
            cls_mask = train_y == classid
            train_x_cls = train_x[cls_mask]
            mean = np.mean(train_x_cls, axis=0)
            std = np.std(train_x_cls, axis=0)
            self.mean_by_classid[classid] = mean
            self.std_by_classid[classid] = std
            self.class_freqs[classid] = np.mean(cls_mask)

    def predict(self, x):
        """Predict the model.

        Arguments:
            x : nsamples x nfeatures

        Returns:
            preds : nsamples
        """
        if not self.fitted:
            raise

        posterior_by_classid = {}
        for classid in self.classids:
            mean = self.mean_by_classid[classid]
            std = self.std_by_classid[classid]
            likelihoods = gaussian_pdf(x, mean, std)
            prior = self.class_freqs[classid]
            posterior = np.prod(likelihoods, axis=1) * prior
            posterior_by_classid[classid] = posterior
        probs = stack_posterior(posterior_by_classid, self.classids)
        preds = np.argmax(probs, axis=1)
        return preds


class MultinomialNB(object):
    def __init__(self):
        self.fitted = False

    def fit(self, train_x, train_y):
        """Fit the model.

        Arguments:
            train_x : nsamples x nfeatures
            train_y : nsamples
        """
        if self.fitted:
            raise
        self.fitted = True

        self.classids = sorted(set(train_y))
        self.feature_freqs = {}
        self.class_freqs = {}
        for classid in self.classids:
            cls_mask = train_y == classid
            train_x_cls = train_x[cls_mask]
            self.feature_freqs[classid] = np.log(
                np.sum(train_x_cls, axis=0) / train_x_cls.sum() # TODO: add laplace smoothing
            )
            self.class_freqs[classid] = np.log(
                np.mean(cls_mask)
            )

    def predict(self, x):
        """Predict the model.

        Arguments:
            x : nsamples x nfeatures

        Returns:
            preds : nsamples
        """
        if not self.fitted:
            raise

        posterior_by_classid = {}
        for classid in self.classids:
            likelihoods = x @ self.feature_freqs[classid]
            prior = self.class_freqs[classid]
            posterior = likelihoods + prior
            posterior_by_classid[classid] = posterior
        probs = stack_posterior(posterior_by_classid, self.classids)
        preds = np.argmax(probs, axis=1)
        return preds


def onehot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b


def sigmoid(a):
    return 1 / (1+np.exp(-a))


class LogisticRegression(object):
    def __init__(self, lr=0.001, steps=5000):
        self.fitted = False
        self.lr = lr
        self.steps = steps

    def fit(self, train_x, train_y):
        """Fit the model.

        Arguments:
            train_x : nsamples x nfeatures
            train_y : nsamples
        """
        if self.fitted:
            raise
        self.fitted = True

        self.scaler = StandardScaler()
        self.scaler.fit(train_x)
        train_x = self.scaler.transform(train_x)

        x = train_x
        yhat = onehot(train_y) # nsamples x noutputs

        nfeatures = x.shape[1]
        noutputs = yhat.shape[1]
        self.w = np.random.randn(nfeatures, noutputs)
        self.b = np.zeros((noutputs))

        for i in range(self.steps):

            # [nsamples, noutputs] = [nsamples, nfeatures] * [nfeatures, noutputs] + [noutputs]
            a = x @ self.w + self.b
            y = sigmoid(a)
            L = -yhat * np.log(y)

            if i % 200 == 0:
                print(f'iter {i}, loss {L.mean()}')

            dLdy = -yhat / y

            dyda = y * (1 - y)
            dLda = dLdy * dyda

            # [nsamples, nfeatures]
            dadw = x
            # [nfeatures, noutputs] = [nsamples, nfeatures].T * [nsamples, noutputs]
            dLdw = dadw.T @ dLda

            dadb = 1
            # [nsample, noutputs]
            dLdb = dLda * dadb
            # [noutputs]
            dLdb = np.mean(dLdb, axis=0)

            # [nfeatures, noutputs]
            dadx = self.w
            # [nsamples, nfeatures] = [nsamples, noutputs] * [nfeatures, noutputs].T
            dLdx = dLda @ dadx.T

            # TODO: add l2 regularization
            self.w = self.w - (self.lr * dLdw)
            self.b = self.b - (self.lr * dLdb)

    def predict(self, x):
        """Predict the model.

        Arguments:
            x : nsamples x nfeatures

        Returns:
            preds : nsamples
        """
        if not self.fitted:
            raise

        x = self.scaler.transform(x)

        y = x @ self.w + self.b
        preds = np.argmax(y, axis=1)
        return preds


def accuracy(gt, pred):
    return (gt==pred).mean()


def main():
    train_x, train_y, val_x, val_y = load_classification_data()

    for nb in [
            GaussianNB(),
            SklearnGaussianNB(),
            MultinomialNB(),
            SklearnMultinomialNB(),
            LogisticRegression(),
            SklearnLogisticRegression(),
    ]:
        print('x'*50)

        print(nb)
        nb.fit(train_x, train_y)

        train_yp = nb.predict(train_x)
        print('train accuracy', accuracy(train_y, train_yp))

        val_yp = nb.predict(val_x)
        print('val accuracy', accuracy(val_y, val_yp))

        print('y'*50)


if __name__ == '__main__':
    main()
