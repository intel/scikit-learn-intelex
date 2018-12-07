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

# fall back to numpy genfromtxt
def np_read_csv(f, c, s=0, n=np.iinfo(np.int64).max, t=np.float64):
    if s==0 and n==np.iinfo(np.int64).max:
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2, dtype=t)
    a = np.genfromtxt(f, usecols=c, delimiter=',', skip_header=s, max_rows=n, dtype=t)
    if a.shape[0] == 0:
        raise Exception("done")
    if a.ndim == 1:
        return a[:, np.newaxis]
    return a

pd_read_csv = lambda f, c, s=0, n=None, t=np.float64: pd.read_csv(f, usecols=c, delimiter=',', header=None, skiprows=s, nrows=n, dtype=t)

csr_read_csv = lambda f, c, s=0, n=None, t=np.float64: csr_matrix(np_read_csv(f, c, s=s, n=n, t=t))


class TestExNpy(unittest.TestCase):

    def setUp(self):
        global read_csv
        self.read_csv = np_read_csv

    def test_adagrad_mse_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "adagrad_mse_batch.csv"), range(1))
        import adagrad_mse_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_correlation_distance_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "correlation_distance_batch.csv"), range(1))
        import correlation_distance_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        r = result.correlationDistance
        self.assertTrue(np.allclose(np.array([[np.amin(r)],[np.amax(r)],[np.mean(r)],[np.average(r)]]), testdata))

    def test_cosine_distance_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "cosine_distance_batch.csv"), range(1))
        import cosine_distance_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        r = result.cosineDistance
        self.assertTrue(np.allclose(np.array([[np.amin(r)],[np.amax(r)],[np.mean(r)],[np.average(r)]]), testdata))

    def test_kdtree_knn_classification_batch(self):
        import kdtree_knn_classification_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, test_labels) = ex.main()
        self.assertTrue(np.count_nonzero(test_labels != predict_result.prediction) < 170)

    @unittest.skipIf(daal_version < (2019, 1), "not supported in this library version")
    def test_decision_forest_classification_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_forest_classification_batch.csv"), range(1))
        import decision_forest_classification_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    @unittest.skipIf(daal_version < (2019, 1), "not supported in this library version")
    def test_decision_forest_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_forest_regression_batch.csv"), range(1))
        import decision_forest_regression_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_decision_tree_classification_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_tree_classification_batch.csv"), range(1))
        import decision_tree_classification_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_decision_tree_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "decision_tree_regression_batch.csv"), range(1))
        import decision_tree_regression_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_gradient_boosted_classification_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "gradient_boosted_classification_batch.csv"), range(1))
        import gradient_boosted_classification_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_gradient_boosted_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "gradient_boosted_regression_batch.csv"), range(1))
        import gradient_boosted_regression_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        #MSE
        self.assertTrue(np.square(predict_result.prediction - testdata).mean() < 1e-2)

    def test_kmeans_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "kmeans_batch.csv"), range(20))
        import kmeans_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.centroids, testdata))

    def test_lbfgs_cr_entr_loss_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "lbfgs_cr_entr_loss_batch.csv"), range(1))
        import lbfgs_cr_entr_loss_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_lbfgs_mse_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "lbfgs_mse_batch.csv"), range(1))
        import lbfgs_mse_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_linear_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "linear_regression_batch.csv"), range(2))
        import linear_regression_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_linear_regression_stream(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "linear_regression_batch.csv"), range(2))
        import linear_regression_streaming as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_ridge_regression_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "ridge_regression_batch.csv"), range(2))
        import ridge_regression_batch as ex
        ex.read_csv = self.read_csv
        (predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_ridge_regression_stream(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "ridge_regression_batch.csv"), range(2))
        import ridge_regression_streaming as ex
        ex.read_csv = self.read_csv
        (predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_log_reg_binary_dense_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "log_reg_binary_dense_batch.csv"), range(1))
        import log_reg_binary_dense_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_log_reg_dense_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "log_reg_dense_batch.csv"), range(1))
        import log_reg_dense_batch as ex
        ex.read_csv = self.read_csv
        (_, predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_low_order_moms_dense_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "low_order_moms_dense_batch.csv"), range(10))
        import low_order_moms_dense_batch as ex
        ex.read_csv = self.read_csv
        res = ex.main()
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
        ex.read_csv = self.read_csv
        res = ex.main()
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
        ex.read_csv = self.read_csv
        ( _,result) = ex.main()
        self.assertTrue(np.allclose(result.weights, testdata))

    def test_naive_bayes_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "naive_bayes_batch.csv"), range(1))
        import naive_bayes_batch as ex
        ex.read_csv = self.read_csv
        (predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_naive_bayes_stream(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "naive_bayes_batch.csv"), range(1))
        import naive_bayes_streaming as ex
        ex.read_csv = self.read_csv
        (predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_cholesky_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "cholesky_batch.csv"), range(5))
        import cholesky_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.choleskyFactor, testdata))

    def test_pca_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "pca_batch.csv"), range(10))
        import pca_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.eigenvectors, testdata))

    def test_pca_transform_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "pca_transform_batch.csv"), range(2))
        import pca_transform_batch as ex
        ex.read_csv = self.read_csv
        _, result = ex.main()
        self.assertTrue(np.allclose(result.transformedData, testdata))

    def test_sgd_logistic_loss_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "sgd_logistic_loss_batch.csv"), range(1))
        import sgd_logistic_loss_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_sgd_mse_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "sgd_mse_batch.csv"), range(1))
        import sgd_mse_batch as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_svd_batch(self):
        import svd_batch as ex
        ex.read_csv = self.read_csv
        (data, result) = ex.main()
        self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix,np.diag(result.singularValues[0])),result.rightSingularMatrix)))

    def test_svd_stream(self):
        import svd_streaming as ex
        ex.read_csv = self.read_csv
        result = ex.main()
        data = np.loadtxt("./data/distributed/svd_1.csv", delimiter=',')
        for f in ["./data/distributed/svd_{}.csv".format(i) for i in range(2,5)]:
            data = np.append(data, np.loadtxt(f, delimiter=','), axis=0)
        self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix,np.diag(result.singularValues[0])),result.rightSingularMatrix)))

    def test_svm_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "svm_batch.csv"), range(1))
        import svm_batch as ex
        ex.read_csv = self.read_csv
        (predict_result, _) = ex.main()
        self.assertTrue(np.absolute(predict_result.prediction - testdata).max() < np.absolute(predict_result.prediction.max() - predict_result.prediction.min()) * 0.05)

    def test_svm_multiclass_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "svm_multiclass_batch.csv"), range(1))
        import svm_multiclass_batch as ex
        ex.read_csv = self.read_csv
        (predict_result, _) = ex.main()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_univariate_outlier_batch(self):
        testdata = np_read_csv(os.path.join(unittest_data_path, "univariate_outlier_batch.csv"), range(1))
        import univariate_outlier_batch as ex
        ex.read_csv = self.read_csv
        ( _,result) = ex.main()
        self.assertTrue(np.allclose(result.weights, testdata))


class TestExPd(TestExNpy):
    def setUp(self):
        self.read_csv = pd_read_csv

class TestExCSR(TestExNpy):
    def setUp(self):
        self.read_csv = csr_read_csv


if __name__ == '__main__':
    unittest.main()
