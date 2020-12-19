# coding: utf-8
import os
import sys
import operator

qoi_list = ['failed', 'passed', 'xpassed', 'xfailed', 'skipped', 'deselected']
def get_counts(tkn):
    data = [x.split() for x in os.getenv(tkn).split(',')]
    counts = {x[-1]: int(x[-2]) for x in data if len(x) > 0 and x[-1] in qoi_list}
    return counts


def sum_of_attributes(counts, keys):
    if isinstance(keys, str):
        return counts.get(keys, 0)
    return sum([ counts.get(k, 0) for k in keys])


def comp(k, op):
    return op(sum_of_attributes(d4p, k), sum_of_attributes(skl, k))


d4p, skl = get_counts('D4P'), get_counts('SKL')

if d4p and skl and all((
        comp('failed', operator.le), # patched has <= fails
        comp(['passed', 'skipped'], operator.ge), # patched run may have >= passes due to extra tests discovered
                                                  # but some test may have been skipped due to network intermittent issues
        comp('xpassed', operator.eq),
        comp('xfailed', operator.eq),
    )):
    print("Patched scikit-learn passed the compatibility check")
    sys.exit(0)
else:
    print("Patched run: {}".format(d4p))
    print("Unpatched run: {}".format(skl))
    sys.exit(1)
