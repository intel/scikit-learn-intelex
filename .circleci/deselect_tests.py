# coding: utf-8
import argparse
import os.path
from yaml import load as yaml_load

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        prog="deselect_tests.py",
        description="Produce pytest CLI options to deselect tests specified in yaml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argParser.add_argument('conf_file', nargs=1, type=str)
    argParser.add_argument('--absolute', action='store_true')

    args = argParser.parse_args()

    fn = args.conf_file[0] 
    if os.path.exists(fn):
        with open(fn, 'r') as fh:
            dt = yaml_load(fh)

        if args.absolute:
            import sklearn
            base_dir = os.path.relpath(os.path.dirname(sklearn.__file__), os.path.expanduser('~')) + '/'
        else:
            base_dir = ""

        print(" ".join([ "--deselect " + base_dir + test_name for test_name in dt.get('deselected_tests', []) ]))
