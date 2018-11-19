#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
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

from collections import namedtuple, defaultdict

pydefaults = defaultdict(lambda: None)
pydefaults.update({'double': 'NaN64',
                 'float': 'NaN32',
                 'int': '-1',
                 'long': '-1',
                 'size_t': '-1',
                 'bool': 'False',
                 #'std::string' : '""',
                 'std::string &' : '""',
})

cppdefaults = defaultdict(lambda: 'NULL')
cppdefaults.update({'bool': 'false',})


FmtVar = namedtuple('formatted_variable',
                    ['name',      # variable's name
                     'typ',       # type
                     'default',   # default value
                     'arg_cpp',   # use as argument to a C++ function call from C++
                     'arg_cy',    # use as argument to a C++ function call in cython
                     'arg_py',    # use as argument to a Python function call in cython
                     'decl_dflt_cpp',  # use as var declaration in C++ with default value
                     'decl_cpp',  # use as var declaration in C++ without default
                     'decl_dflt_cy',   # use as C++ member/var declaration in Cython with default
                     'decl_dflt_py',   # use as Python member/var declaration in Cython with default
                     'decl_member',# use as member declaration in C++
                     'arg_member',# use as member used in C++
                     'init_member', # initializer for member var
                    ])
FmtVar.__new__.__defaults__ = ('',) * len(FmtVar._fields)

def mk_var(decl):
    if decl == '':
        return FmtVar()
    d          = decl.rsplit('=', 1)
    t, name    = d[0].strip().rsplit(' ', 1)
    const      = 'const ' if 'const' in t else ''
    ref        = '&' if any('&' in x for x in [t, name]) else ''
    ptr        = '*' if any('*' in x for x in [t, name]) else ''
    name       = name.replace('&', '').replace('*', '').strip()
    typ        = t.replace('const', '').replace('&', '').replace('*', '').strip()
    pydefault  = ' = {}'.format(pydefaults[typ]) if len(d) > 1 else ''
    cppdefault = ' = {}'.format(cppdefaults[typ]) if len(d) > 1 else ''
    assert(' ' not in typ), 'Error in parsing variable "{}"'.format(decl)

    return FmtVar(name          = name,
                  typ           = typ,
                  arg_cpp       = name,
                  arg_cy        = name,
                  arg_py        = name,
                  decl_dflt_cpp = '{}{}{} {}{}'.format(const, typ, ref if ref != '' else ptr, name, cppdefault),
                  decl_cpp      = '{}{}{} {}'.format(const, typ, ref if ref != '' else ptr, name),
                  decl_dflt_cy  = '{}{} {}{}'.format(typ, ref if ref != '' else ptr, name, pydefault),
                  decl_dflt_py  = '{} {}{}'.format(typ, name, pydefault),
                  decl_member   = '{}{} _{}'.format(typ, ref if ref != '' else ptr, name),
                  arg_member    = '_{}'.format(name),
                  init_member   = '_{0}({0})'.format(name),
    )

                  
