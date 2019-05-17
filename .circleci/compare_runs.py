# coding: utf-8
import os
import sys
import operator

qoi_list = ['failed', 'passed', 'xpassed', 'xfailed', 'skipped', 'deselected']
def get_counts(tkn):
    data = [x.split() for x in os.getenv(tkn).split(',')]
    counts = {x[-1]: int(x[-2]) for x in data if len(x) > 0 and x[-1] in qoi_list}
    return counts

def comp(k, op):
    return op(d4p.get(k, 0), skl.get(k, 0))


d4p, skl = get_counts('D4P'), get_counts('SKL')

if d4p and skl and all((
        comp('failed', operator.le), # patched has <= fails
        comp('passed', operator.ge), # patches has >= passes due to extra tests discovered
        comp('xpassed', operator.eq),
        comp('xfailed', operator.eq),
    )):
    print("Patched scikit-learn passed the compatibility check")
    sys.exit(0)
else:
    print("Patched run: {}".format(d4p))
    print("Unpatched run: {}".format(skl))
    sys.exit(1)
