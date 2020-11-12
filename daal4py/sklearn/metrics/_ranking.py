import daal4py as d4p

def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None):
    return d4p.daal_roc_auc_score(y_true, y_score)