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


class TestExNpyArray(unittest.TestCase):
    "We run and validate all the examples but read data with numpy, so working natively on a numpy arrays"

    def call(self, ex):
        return ex.main(readcsv=np_read_csv)

    def test_adagrad_mse_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "adagrad_mse_batch.csv"))
        import adagrad_mse_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_association_rules_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "association_rules_batch.csv"))
        import association_rules_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.confidence, testdata))

    def test_correlation_distance_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "correlation_distance_batch.csv"))
        import correlation_distance_batch as ex
        result = self.call(ex)
        r = result.correlationDistance
        self.assertTrue(np.allclose(np.array([[np.amin(r)],[np.amax(r)],[np.mean(r)],[np.average(r)]]), testdata))

    def test_cosine_distance_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "cosine_distance_batch.csv"), range(1))
        import cosine_distance_batch as ex
        result = self.call(ex)
        r = result.cosineDistance
        self.assertTrue(np.allclose(np.array([[np.amin(r)],[np.amax(r)],[np.mean(r)],[np.average(r)]]), testdata))

    def test_covariance_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "covariance.csv"))
        import covariance_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.covariance, testdata))

    def test_covariance_streaming(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "covariance.csv"))
        import covariance_streaming as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.covariance, testdata))

    def test_kdtree_knn_classification_batch(self):
        import kdtree_knn_classification_batch as ex
        (_, predict_result, test_labels) = self.call(ex)
        self.assertTrue(np.count_nonzero(test_labels != predict_result.prediction) < 170)

    @unittest.skipIf(daal_version < (2019, 1), "not supported in this library version")
    def test_decision_forest_classification_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_forest_classification_batch.csv"), range(1))
        import decision_forest_classification_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    @unittest.skipIf(daal_version < (2019, 1), "not supported in this library version")
    def test_decision_forest_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_forest_regression_batch.csv"), range(1))
        import decision_forest_regression_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_decision_tree_classification_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_tree_classification_batch.csv"), range(1))
        import decision_tree_classification_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_decision_tree_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_tree_regression_batch.csv"), range(1))
        import decision_tree_regression_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_gradient_boosted_classification_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "gradient_boosted_classification_batch.csv"), range(1))
        import gradient_boosted_classification_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_gradient_boosted_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "gradient_boosted_regression_batch.csv"), range(1))
        import gradient_boosted_regression_batch as ex
        (_, predict_result, _) = self.call(ex)
        #MSE
        self.assertTrue(np.square(predict_result.prediction - testdata).mean() < 1e-2)

    def test_kmeans_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "kmeans_batch.csv"), range(20))
        import kmeans_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.centroids, testdata))

    def test_lbfgs_cr_entr_loss_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "lbfgs_cr_entr_loss_batch.csv"), range(1))
        import lbfgs_cr_entr_loss_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_lbfgs_mse_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "lbfgs_mse_batch.csv"), range(1))
        import lbfgs_mse_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_linear_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "linear_regression_batch.csv"), range(2))
        import linear_regression_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_linear_regression_stream(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "linear_regression_batch.csv"), range(2))
        import linear_regression_streaming as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_ridge_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "ridge_regression_batch.csv"), range(2))
        import ridge_regression_batch as ex
        (predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_ridge_regression_stream(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "ridge_regression_batch.csv"), range(2))
        import ridge_regression_streaming as ex
        (predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_log_reg_binary_dense_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "log_reg_binary_dense_batch.csv"), range(1))
        import log_reg_binary_dense_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_log_reg_dense_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "log_reg_dense_batch.csv"), range(1))
        import log_reg_dense_batch as ex
        (_, predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_low_order_moms_dense_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "low_order_moms_dense_batch.csv"), range(10))
        import low_order_moms_dense_batch as ex
        res = self.call(ex)
        r = np.vstack((
            res.minimum,
            res.maximum,
            res.sum,
            res.sumSquares,
            res.sumSquaresCentered,
            res.mean,
            res.secondOrderRawMoment,
            res.variance,
            res.standardDeviation,
            res.variation
        ))
        self.assertTrue(np.allclose(r, testdata))

    def test_low_order_moms_dense_stream(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "low_order_moms_dense_batch.csv"), range(10))
        import low_order_moms_streaming as ex
        res = self.call(ex)
        r = np.vstack((
            res.minimum,
            res.maximum,
            res.sum,
            res.sumSquares,
            res.sumSquaresCentered,
            res.mean,
            res.secondOrderRawMoment,
            res.variance,
            res.standardDeviation,
            res.variation
        ))
        self.assertTrue(np.allclose(r, testdata))

    def test_multivariate_outlier_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "multivariate_outlier_batch.csv"), range(1))
        import multivariate_outlier_batch as ex
        ( _,result) = self.call(ex)
        self.assertTrue(np.allclose(result.weights, testdata))

    def test_naive_bayes_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "naive_bayes_batch.csv"), range(1))
        import naive_bayes_batch as ex
        (predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_naive_bayes_stream(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "naive_bayes_batch.csv"), range(1))
        import naive_bayes_streaming as ex
        (predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_cholesky_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "cholesky_batch.csv"), range(5))
        import cholesky_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.choleskyFactor, testdata))

    def test_pca_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "pca_batch.csv"), range(10))
        import pca_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.eigenvectors, testdata))

    def test_pca_transform_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "pca_transform_batch.csv"), range(2))
        import pca_transform_batch as ex
        _, result = self.call(ex)
        self.assertTrue(np.allclose(result.transformedData, testdata))

    def test_sgd_logistic_loss_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "sgd_logistic_loss_batch.csv"), range(1))
        import sgd_logistic_loss_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_sgd_mse_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "sgd_mse_batch.csv"), range(1))
        import sgd_mse_batch as ex
        result = self.call(ex)
        self.assertTrue(np.allclose(result.minimum, testdata))

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

    def test_svm_multiclass_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "svm_multiclass_batch.csv"), range(1))
        import svm_multiclass_batch as ex
        (predict_result, _) = self.call(ex)
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_univariate_outlier_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "univariate_outlier_batch.csv"), range(1))
        import univariate_outlier_batch as ex
        ( _,result) = self.call(ex)
        self.assertTrue(np.allclose(result.weights, testdata))


class TestExPandasDF(TestExNpyArray):
    "We run and validate all the examples but read data with pandas, so working natively on a pandas DataFrame"
    def call(self, ex):
        return ex.main(readcsv=pd_read_csv)


class TestExCSRMatrix(TestExNpyArray):
    """
    We run and validate all the examples but use scipy-sparse-csr_matrix as input data.
    We also let algos use CSR method (some algos ignore the method argument since they do not specifically support CSR).
    """
    def call(self, ex):
        method = 'singlePassCSR' if any(x in ex.__name__ for x in ['low_order_moms', 'covariance']) else 'fastCSR'
        if hasattr(ex, 'dflt_method'):
            low_order_moms
            method = ex.dflt_method.replace('defaultDense', 'fastCSR').replace('Dense', 'CSR')
        return ex.main(readcsv=csr_read_csv, method=method)


if __name__ == '__main__':
    unittest.main()
