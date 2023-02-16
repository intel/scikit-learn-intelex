import dpctl

q = dpctl.SyclQueue("gpu")

ns, nf = 1024, 16

from sklearn.datasets import make_regression

X, y = make_regression(n_samples = ns, n_features = nf)

from sklearn.model_selection import train_test_split

X, Xt, y, yt = train_test_split(X, y)

from onedal.linear_model import LinearRegression

m = LinearRegression(fit_intercept = True, copy_X = False)

m.fit(X, y, queue = q)

print(Xt.shape)

res = m.predict(Xt, queue = q)

print(res, yt)
