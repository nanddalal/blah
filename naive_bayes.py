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


def accuracy(gt, pred):
    return (gt==pred).mean()


def main():
    train_x, train_y, val_x, val_y = load_classification_data()

    for nb in [GaussianNB(), SklearnGaussianNB(), MultinomialNB(), SklearnMultinomialNB()]:
        print(nb)
        nb.fit(train_x, train_y)

        train_yp = nb.predict(train_x)
        print('train accuracy', accuracy(train_y, train_yp))

        val_yp = nb.predict(val_x)
        print('val accuracy', accuracy(val_y, val_yp))


if __name__ == '__main__':
    main()
