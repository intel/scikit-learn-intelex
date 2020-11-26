#
#*****************************************************************************
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************

import daal4py as d4p
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import label_binarize
from sklearn.metrics._ranking import _multiclass_roc_auc_score, _binary_roc_auc_score
from sklearn.metrics._base import _average_binary_score
from .._utils import get_patch_message
import logging
from functools import partial

def _daal_roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None):
    y_type = type_of_target(y_true)
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    if y_type == "multiclass" or (y_type == "binary" and
                                  y_score.ndim == 2 and
                                  y_score.shape[1] > 2):
        # do not support partial ROC computation for multiclass
        if max_fpr is not None and max_fpr != 1.:
            raise ValueError("Partial AUC computation not available in "
                             "multiclass setting, 'max_fpr' must be"
                             " set to `None`, received `max_fpr={0}` "
                             "instead".format(max_fpr))
        if multi_class == 'raise':
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("sklearn"))
        return _multiclass_roc_auc_score(y_true, y_score, labels,
                                         multi_class, average, sample_weight)
    elif y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, classes=labels)[:, 0]
        if max_fpr is None and sample_weight is None and len(labels) == 2:
            logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("daal"))
            return d4p.daal_roc_auc_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
        logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("sklearn"))
        return _average_binary_score(partial(_binary_roc_auc_score,
                                             max_fpr=max_fpr),
                                     y_true, y_score, average,
                                     sample_weight=sample_weight)
    else:  # multilabel-indicator
        logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("sklearn"))
        return _average_binary_score(partial(_binary_roc_auc_score,
                                             max_fpr=max_fpr),
                                     y_true, y_score, average,
                                     sample_weight=sample_weight)
