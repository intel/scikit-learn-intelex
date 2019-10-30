#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
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
#******************************************************************************/

import os
import platform
import struct
import subprocess
import sys

from daal4py import __daal_link_version__ as dv, __has_dist__
daal_version = tuple(map(int, (dv[0:4], dv[4:8])))

from os.path import join as jp
from time import gmtime, strftime
from collections import defaultdict

exdir = os.path.dirname(os.path.realpath(__file__))

IS_WIN = platform.system() == 'Windows'

assert 8 * struct.calcsize('P') in [32, 64]

if 8 * struct.calcsize('P') == 32:
    logdir = jp(exdir, '_results', 'ia32')
else:
    logdir = jp(exdir, '_results', 'intel64')

req_version = defaultdict(lambda:(2019,0))
req_version['decision_forest_classification_batch.py'] = (2019,1)
req_version['decision_forest_regression_batch.py'] = (2019,1)
req_version['adaboost_batch.py'] = (2022,0)
req_version['brownboost_batch.py'] = (2022,0)
req_version['logitboost_batch.py'] = (2022,0)
req_version['stump_classification_batch.py'] = (2022,0)
req_version['stump_regression_batch.py'] = (2022,0)
req_version['saga_batch.py'] = (2019,3)
req_version['dbscan_batch.py'] = (2019,5)
req_version['lasso_regression_batch.py'] = (2019,5)

def get_exe_cmd(ex, nodist, nostream):
    if req_version[os.path.basename(ex)] > daal_version:
        return None
    if any(ex.endswith(x) for x in ['batch.py', 'stream.py']):
        return '"' + sys.executable + '" "' + ex + '"'
    if not nostream and ex.endswith('streaming.py'):
        return '"' + sys.executable + '" "' + ex + '"'
    if not nodist and ex.endswith('spmd.py'):
        if IS_WIN:
            return 'mpiexec -localonly -n 4 "' + sys.executable + '" "' + ex + '"'
        else:
            return 'mpirun -n 4 "' + sys.executable + '" "' + ex + '"'
    return None

def run_all(nodist=False, nostream=False):
    success = 0
    n = 0
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    for (dirpath, dirnames, filenames) in os.walk(exdir):
        for script in filenames:
            if any(script.endswith(x) for x in ['spmd.py', 'streaming.py', 'stream.py', 'batch.py']):
                n += 1
                logfn = jp(logdir, script.replace('.py', '.res'))
                with open(logfn, 'w') as logfile:
                    print('\n##### ' + jp(dirpath, script))
                    execute_string = get_exe_cmd(jp(dirpath, script), nodist, nostream)
                    if execute_string:
                        proc = subprocess.Popen(execute_string if IS_WIN else ['/bin/bash', '-c', execute_string],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT,
                                                shell=(True if IS_WIN else False))
                        out = proc.communicate()[0]
                        logfile.write(out.decode('ascii'))
                        if proc.returncode:
                            print(out)
                            print(strftime("%H:%M:%S", gmtime()) + '\tFAILED\t' + script + '\twith errno\t' + str(proc.returncode))
                        else:
                            success += 1
                            print(strftime("%H:%M:%S", gmtime()) + '\tPASSED\t' + script)
                    else:
                        success += 1
                        print(strftime("%H:%M:%S", gmtime()) + '\tSKIPPED\t' + script)

    if success != n:
        print('{}/{} examples passed/skipped, {} failed'.format(success,n, n - success))
        print('Error(s) occured. Logs can be found in ' + logdir)
        return 4711
    else:
        print('{}/{} examples passed/skipped'.format(success,n))
        return 0

if __name__ == '__main__':
    sys.exit(run_all('nodist' in sys.argv or not __has_dist__, 'nostream' in sys.argv))
