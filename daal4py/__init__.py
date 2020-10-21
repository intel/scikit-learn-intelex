try:
    from _daal4py import *
    from _daal4py import _get__version__, _get__daal_link_version__, _get__daal_run_version__, __has_dist__
except ImportError as e:
    s = str(e)
    if 'libfabric' in s:
        raise ImportError(s + '\n\nActivating your conda environment or sourcing mpivars.[c]sh/psxevars.[c]sh may solve the issue.\n')
    raise

import logging
import warnings
import os
import sys
logLevel = os.environ.get("IDP_SKLEARN_VERBOSE")
try:
    if not logLevel is None:
        logging.basicConfig(stream=sys.stdout, format='%(levelname)s: %(message)s', level=logLevel.upper())
except:
    warnings.warn('Unknown level "{}" for logging.\n'
                    'Please, use one of "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG".'.format(logLevel))
