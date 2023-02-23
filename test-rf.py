import logging as lg
lg.getLogger().setLevel(lg.DEBUG)
from sklearnex import patch_sklearn

patch_sklearn(preview=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=50000, n_features=64)
rfc = RandomForestClassifier(n_estimators=20)
assert 'sklearnex.preview' in rfc.__module__
print(rfc.fit(x, y).score(x, y))
