import numpy as np
import pytest
import random
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE as sklearn_TSNE
from daal4py.sklearn.manifold import TSNE as daal_TSNE

RNG = np.random.RandomState(0)
DIGITS = load_digits()
N_ITER = 1
DIVERG = 1.04


def compare_with_sklearn_tsne(early_exaggeration=12.0, learning_rate=200,
                              min_grad_norm=1e-7, angle=0.5, describe=""):

    x = DIGITS.data

    # Models
    sklearn_model = sklearn_TSNE(early_exaggeration=early_exaggeration,
                                 learning_rate=learning_rate,
                                 min_grad_norm=min_grad_norm, angle=angle)

    daal_model = daal_TSNE(early_exaggeration=early_exaggeration,
                           learning_rate=learning_rate,
                           min_grad_norm=min_grad_norm, angle=angle)

    # Train
    sklearn_model.fit(x)
    daal_model.fit(x)

    # Divergence
    sklearn_divergence = sklearn_model.kl_divergence_
    daal_divergence = daal_model.kl_divergence_
    ratio = daal_divergence / sklearn_divergence
    reason = describe + \
        f"daal_divergence = {daal_divergence}, sklearn_divergence = {sklearn_divergence}"
    assert ratio <= DIVERG, reason

    # N_iter
    sklearn_model_n_iter = sklearn_model.n_iter_
    daal_model_n_iter = daal_model.n_iter_
    ratio = daal_model_n_iter / sklearn_model_n_iter
    reason = describe + \
        f"daal_n_iter = {daal_model_n_iter}, sklearn_n_iter = {sklearn_model_n_iter}"
    assert ratio == N_ITER, reason


EARLY_EXAGGERATION = [5.0, 7.0, 9.0, 12.0, 14.0, 16.0]


@pytest.mark.parametrize('early_exaggeration', EARLY_EXAGGERATION)
def test_early_exaggeration(early_exaggeration):
    compare_with_sklearn_tsne(
        early_exaggeration=early_exaggeration,
        describe="early_exaggeration:"
    )


LEARNING_RATE = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]


@pytest.mark.parametrize('learning_rate', LEARNING_RATE)
def test_learning_rate(learning_rate):
    compare_with_sklearn_tsne(
        learning_rate=learning_rate,
        describe="learning_rate:"
    )


ANGLE = [0.2, 0.35, 0.5, 0.65, 0.8]


@pytest.mark.parametrize('angle', ANGLE)
def test_angle(angle):
    compare_with_sklearn_tsne(
        angle=angle,
        describe="angle:"
    )


MIN_GRAD_NORM = [1e-5, 1e-6, 1e-7, 1e-8]


@pytest.mark.parametrize('min_grad_norm', MIN_GRAD_NORM)
def test_min_grad_norm(min_grad_norm):
    compare_with_sklearn_tsne(
        min_grad_norm=min_grad_norm,
        describe="min_grad_norm:"
    )
