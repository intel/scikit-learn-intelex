import daal4py as d4p

def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None):
    print('qwe')
    return d4p.daal_roc_auc_score(y_true.reshape(1, -1), y_score.reshape(1, -1))