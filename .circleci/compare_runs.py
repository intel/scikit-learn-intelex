# coding: utf-8
import os
import sys
import operator

def get_counts(tkn):
    data = [x.split() for x in os.getenv(tkn).split(',')]
    counts = {x[1]: int(x[0]) for x in data if len(x) == 2}
    return counts

def comp(k, op):
    return op(d4p.get(k, 0), skl.get(k, 0))


d4p, skl = get_counts('D4P'), get_counts('SKL')
if all((comp('failed', operator.eq),
     comp('passed', operator.ge),
     comp('xpassed', operator.eq),
     comp('xfailed', operator.eq))
   ):
    print("Patched scikit-learn passed the compatibility check")
    sys.exit(0)
else:
    print("Patched run: {}".format(d4p))
    print("Unpatched run: {}".format(skl))
    sys.exit(1)
