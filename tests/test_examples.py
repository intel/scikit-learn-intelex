import os
import sys
test_path = os.path.abspath(os.path.dirname(__file__))
unittest_data_path = os.path.join(test_path, "unittest_data")
examples_path = os.path.join(os.path.dirname(test_path), "examples")
sys.path.insert(0, examples_path)
os.chdir(examples_path)

import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from daal4py import __daal_link_version__ as dv
daal_version = tuple(map(int, (dv[0:4], dv[4:8])))

# function reading file and returning numpy array
def np_read_csv(f, c=None, s=0, n=np.iinfo(np.int64).max, t=np.float64):
    if s==0 and n==np.iinfo(np.int64).max:
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2, dtype=t)
    a = np.genfromtxt(f, usecols=c, delimiter=',', skip_header=s, max_rows=n, dtype=t)
    if a.shape[0] == 0:
        raise Exception("done")
    if a.ndim == 1:
        return a[:, np.newaxis]
    return a

# function reading file and returning pandas DataFrame
pd_read_csv = lambda f, c=None, s=0, n=None, t=np.float64: pd.read_csv(f, usecols=c, delimiter=',', header=None, skiprows=s, nrows=n, dtype=t)

# function reading file and returning scipy.sparse.csr_matrix
csr_read_csv = lambda f, c=None, s=0, n=None, t=np.float64: csr_matrix(pd_read_csv(f, c, s=s, n=n, t=t))


def add_test(cls, e, f=None, attr=None, ver=(0,0)):
    import importlib
    @unittest.skipIf(daal_version < ver, "not supported in this library version")
    def testit(self):
        ex = importlib.import_module(e)
        result = self.call(ex)
        if f and attr:
            testdata = np_read_csv(os.path.join(unittest_data_path, f))
            self.assertTrue(np.allclose(attr(result) if callable(attr) else getattr(result, attr), testdata, atol=1e-05))
        else:
            self.assertTrue(True)
    setattr(cls, 'test_'+e, testit)


class Base():
    """
    We also use generic functions to test these, they get added later.
    """

    def test_kdtree_knn_classification_batch(self):
        import kdtree_knn_classification_batch as ex
        (_, predict_result, test_labels) = self.call(ex)
        self.assertTrue(np.count_nonzero(test_labels != predict_result.prediction) < 170)

    def test_gradient_boosted_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "gradient_boosted_regression_batch.csv"), range(1))
        import gradient_boosted_regression_batch as ex
        (_, predict_result, _) = self.call(ex)
        #MSE
        self.assertTrue(np.square(predict_result.prediction - testdata).mean() < 1e-2)

    def test_svd_batch(self):
        import svd_batch as ex
        (data, result) = self.call(ex)
        self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix,np.diag(result.singularValues[0])),result.rightSingularMatrix)))
 
    def test_svd_stream(self):
        import svd_streaming as ex
        result = self.call(ex)
        data = np.loadtxt("./data/distributed/svd_1.csv", delimiter=',')
        for f in ["./data/distributed/svd_{}.csv".format(i) for i in range(2,5)]:
            data = np.append(data, np.loadtxt(f, delimiter=','), axis=0)
        self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix,np.diag(result.singularValues[0])),result.rightSingularMatrix)))

    def test_svm_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "svm_batch.csv"), range(1))
        import svm_batch as ex
        (predict_result, _) = self.call(ex)
        self.assertTrue(np.absolute(predict_result.prediction - testdata).max() < np.absolute(predict_result.prediction.max() - predict_result.prediction.min()) * 0.05)


gen_examples = [
    ('adaboost_batch', None, None, (2019, 2)),
    ('adagrad_mse_batch', 'adagrad_mse_batch.csv', 'minimum'),
    ('association_rules_batch', 'association_rules_batch.csv', 'confidence'),
    ('bacon_outlier_batch', 'multivariate_outlier_batch.csv', lambda r: r[1].weights),
    ('brownboost_batch', None, None, (2019, 2)),
    ('correlation_distance_batch', 'correlation_distance_batch.csv', lambda r: [[np.amin(r.correlationDistance)],
                                                                                [np.amax(r.correlationDistance)],
                                                                                [np.mean(r.correlationDistance)],
                                                                                [np.average(r.correlationDistance)]]),
    ('cosine_distance_batch', 'cosine_distance_batch.csv', lambda r: [[np.amin(r.cosineDistance)],
                                                                      [np.amax(r.cosineDistance)],
                                                                      [np.mean(r.cosineDistance)],
                                                                      [np.average(r.cosineDistance)]]),
    #('gradient_boosted_regression_batch', 'gradient_boosted_regression_batch.csv', lambda x: x[1].prediction),
    ('cholesky_batch', 'cholesky_batch.csv', 'choleskyFactor'),
    ('covariance_batch', 'covariance.csv', 'covariance', (2019, 2)),
    ('covariance_streaming', 'covariance.csv', 'covariance', (2019, 2)),
    ('decision_forest_classification_batch', 'decision_forest_classification_batch.csv', lambda r: r[1].prediction, (2019, 1)),
    ('decision_forest_regression_batch', 'decision_forest_regression_batch.csv', lambda r: r[1].prediction, (2019, 1)),
    ('decision_tree_classification_batch', 'decision_tree_classification_batch.csv', lambda r: r[1].prediction),
    ('decision_tree_regression_batch', 'decision_tree_regression_batch.csv', lambda r: r[1].prediction),
    ('distributions_bernoulli_batch',),
    ('distributions_normal_batch',),
    ('distributions_uniform_batch',),
    ('em_gmm_batch', 'em_gmm.csv', lambda r: r.covariances[0]),
    ('gradient_boosted_classification_batch',),
    ('implicit_als_batch', 'implicit_als_batch.csv', 'prediction'),
    ('kmeans_batch', 'kmeans_batch.csv', 'centroids'),
    ('lbfgs_cr_entr_loss_batch', 'lbfgs_cr_entr_loss_batch.csv', 'minimum'),
    ('lbfgs_mse_batch', 'lbfgs_mse_batch.csv', 'minimum'),
    ('linear_regression_batch', 'linear_regression_batch.csv', lambda r: r[1].prediction),
    ('linear_regression_streaming', 'linear_regression_batch.csv', lambda r: r[1].prediction),
    ('log_reg_binary_dense_batch', 'log_reg_binary_dense_batch.csv', lambda r: r[1].prediction),
    ('log_reg_dense_batch',),
    ('logitboost_batch', None, None, (2019, 2)),
    ('low_order_moms_dense_batch', 'low_order_moms_dense_batch.csv', lambda r: np.vstack((r.minimum,
                                                                                          r.maximum,
                                                                                          r.sum,
                                                                                          r.sumSquares,
                                                                                          r.sumSquaresCentered,
                                                                                          r.mean,
                                                                                          r.secondOrderRawMoment,
                                                                                          r.variance,
                                                                                          r.standardDeviation,
                                                                                          r.variation))),
    ('low_order_moms_streaming', 'low_order_moms_dense_batch.csv', lambda r: np.vstack((r.minimum,
                                                                                        r.maximum,
                                                                                        r.sum,
                                                                                        r.sumSquares,
                                                                                        r.sumSquaresCentered,
                                                                                        r.mean,
                                                                                        r.secondOrderRawMoment,
                                                                                        r.variance,
                                                                                        r.standardDeviation,
                                                                                        r.variation))),
    ('math_abs_batch',),
    ('math_logistic_batch',),
    ('math_relu_batch',),
    ('math_smoothrelu_batch',),
    ('math_softmax_batch',),
    ('math_tanh_batch',),
    ('multivariate_outlier_batch', 'multivariate_outlier_batch.csv', lambda r: r[1].weights),
    ('naive_bayes_batch', 'naive_bayes_batch.csv', lambda r: r[0].prediction),
    ('naive_bayes_streaming', 'naive_bayes_batch.csv', lambda r: r[0].prediction),
    ('normalization_minmax_batch', 'normalization_minmax.csv', 'normalizedData'),
    ('normalization_zscore_batch', 'normalization_zscore.csv', 'normalizedData'),
    ('pca_batch', 'pca_batch.csv', 'eigenvectors'),
    ('pca_transform_batch', 'pca_transform_batch.csv', lambda r: r[1].transformedData),
    ('pivoted_qr_batch', 'pivoted_qr.csv', 'matrixR'),
    ('quantiles_batch', 'quantiles.csv', 'quantiles'),
    ('qr_batch', 'qr.csv', 'matrixR'),
    ('qr_streaming', 'qr.csv', 'matrixR'),
    ('ridge_regression_batch', 'ridge_regression_batch.csv', lambda r: r[0].prediction),
    ('ridge_regression_streaming', 'ridge_regression_batch.csv', lambda r: r[0].prediction),
    ('saga_batch', None, None, (2019, 2)),
    ('sgd_logistic_loss_batch', 'sgd_logistic_loss_batch.csv', 'minimum'),
    ('sgd_mse_batch', 'sgd_mse_batch.csv', 'minimum'),
    ('sorting_batch',),
    ('stump_classification_batch', None, None, (2019, 2)),
    ('stump_regression_batch', None, None, (2019, 2)),
    ('svm_multiclass_batch', 'svm_multiclass_batch.csv', lambda r: r[0].prediction),
    ('univariate_outlier_batch', 'univariate_outlier_batch.csv', lambda r: r[1].weights),
]

for example in gen_examples:
    add_test(Base, *example)
    

class TestExNpyArray(Base, unittest.TestCase):
    """
    We run and validate all the examples but read data with numpy, so working natively on a numpy arrays.
    """
    def call(self, ex):
        return ex.main(readcsv=np_read_csv)


class TestExPandasDF(Base, unittest.TestCase):
    "We run and validate all the examples but read data with pandas, so working natively on a pandas DataFrame"
    def call(self, ex):
        return ex.main(readcsv=pd_read_csv)


class TestExCSRMatrix(Base, unittest.TestCase):
    """
    We run and validate all the examples but use scipy-sparse-csr_matrix as input data.
    We also let algos use CSR method (some algos ignore the method argument since they do not specifically support CSR).
    """
    def call(self, ex):
        # some algos do not support CSR matrices
        if  ex.__name__.startswith('sorting'):
            self.skipTest("not supporting CSR")
        method = 'singlePassCSR' if any(x in ex.__name__ for x in ['low_order_moms', 'covariance']) else 'fastCSR'
        # cannot use fastCSR ofr implicit als; bug in DAAL?
        if 'implicit_als' in ex.__name__:
            method = 'defaultDense'
        if hasattr(ex, 'dflt_method'):
            low_order_moms
            method = ex.dflt_method.replace('defaultDense', 'fastCSR').replace('Dense', 'CSR')
        return ex.main(readcsv=csr_read_csv, method=method)


if __name__ == '__main__':
    unittest.main()
