# coding: utf-8
import argparse
import os.path
from yaml import FullLoader, load as yaml_load
from distutils.version import LooseVersion
import sklearn
from sklearn import __version__ as sklearn_version
import warnings


def evaluate_cond(cond, v):
    if cond.startswith(">="):
        return LooseVersion(v) >= LooseVersion(cond[2:])
    if cond.startswith("<="):
        return LooseVersion(v) <= LooseVersion(cond[2:])
    if cond.startswith("=="):
        return LooseVersion(v) == LooseVersion(cond[2:])
    if cond.startswith("!="):
        return LooseVersion(v) != LooseVersion(cond[2:])
    if cond.startswith("<"):
        return LooseVersion(v) < LooseVersion(cond[1:])
    if cond.startswith(">"):
        return LooseVersion(v) > LooseVersion(cond[1:])
    warnings.warn(
        'Test selection condition "{0}" should start with '
        '>=, <=, ==, !=, < or > to compare to version of scikit-learn run. '
        'The test will not be deselected'.format(cond))
    return False


def filter_by_version(entry, sk_ver):
    if not entry:
        return None
    t = entry.split(' ')
    if len(t) == 1:
        return entry
    if len(t) != 2:
        return None
    test_name, cond = t
    conds = cond.split(',')
    if all([evaluate_cond(cond, sk_ver) for cond in conds]):
        return test_name
    return None


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        prog="deselect_tests.py",
        description="Produce pytest CLI options to deselect tests specified in yaml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argParser.add_argument('conf_file', nargs=1, type=str)
    argParser.add_argument('--absolute', action='store_true')
    argParser.add_argument('--reduced', action='store_true')
    argParser.add_argument('--public', action='store_true')
    args = argParser.parse_args()

    fn = args.conf_file[0]
    if os.path.exists(fn):
        with open(fn, 'r') as fh:
            dt = yaml_load(fh, Loader=FullLoader)

        if args.absolute:
            base_dir = os.path.relpath(
                os.path.dirname(sklearn.__file__),
                os.path.expanduser('~')) + '/'
        else:
            base_dir = ""

        filtered_deselection = [
            filter_by_version(test_name, sklearn_version)
            for test_name in dt.get('deselected_tests', [])]
        if args.reduced:
            filtered_deselection.extend(
                [filter_by_version(test_name, sklearn_version)
                 for test_name in dt.get('reduced_tests', [])])
        if args.public:
            filtered_deselection.extend(
                [filter_by_version(test_name, sklearn_version)
                 for test_name in dt.get('public', [])])
        pytest_switches = ["--deselect " + base_dir + test_name
                           for test_name in filtered_deselection if test_name]
        print(" ".join(pytest_switches))
