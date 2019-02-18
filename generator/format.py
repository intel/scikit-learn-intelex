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

# Provides helper class to format variables (input args, template args and parameters)
# Different syntax for C++, cython and python are precomputed.
# wrapper_gen uses the precomputed attributes in jinja2 macros

from collections import namedtuple, defaultdict
import re

# default values of paramters/inputs are set by daal itself.
# We indicate with these defaults that we want to use daal's defaults
pydefaults = defaultdict(lambda: 'None')
pydefaults.update({'double': 'NaN64',
                   'float': 'NaN32',
                   'int': '-1',
                   'long': '-1',
                   'size_t': '-1',
                   'bool': 'False',
                   'std::string' : '',
                   #'std::string &' : '""',
})

# Same but when calling C++
# We actually only need bool for distributed/streaming; the rest is handled in use_default
cppdefaults = defaultdict(lambda: 'NULL')
cppdefaults.update({'bool': 'false',})


def flat(typ):
    '''Flatten C++ name, leaving only what's needed to disambiguate names.
       E.g. stripping of leading namespaces and replaceing :: with _
    '''
    typ = typ.replace('daal::algorithms::kernel_function::KernelIfacePtr', 'daal::services::SharedPtr<kernel_function::KernelIface>')
    typ = re.sub(r'(daal::)?(algorithms::)?(engines::)?EnginePtr', r'daal::services::SharedPtr<engines::BatchBase>', typ)
    typ = re.sub(r'(?:daal::)?(?:algorithms::)?([^:]+::)BatchPtr', r'daal::services::SharedPtr<\1Batch>', typ)
    typ = re.sub(r'(daal::)?services::SharedPtr<([^>]+)>', r'\2__iface__', typ)
    nn = typ.split('::')
    if nn[0] == 'daal':
        if nn[1] == 'algorithms':
            r = '_'.join(nn[2:])
        else:
            r = '_'.join(nn[1:])
    elif nn[0] == 'algorithms':
        r = '_'.join(nn[1:])
    else:
        r = '_'.join(nn)
    return r

def cy_callext(arg, typ_cy, typ_cyext, s2e=None):
    '''Where needed decorate argument with conversion when calling C++ from cython'''
    if 'data_or_file' in typ_cy:
        return 'mk_data_or_file({0})'.format(arg)
    if 'dict_numerictable' in typ_cy:
        return 'make_dnt(<PyObject *>' + arg + ((', ' + s2e) if s2e else '') + ')'
    if 'list_numerictable' in typ_cy:
        return 'make_datacoll(<PyObject *>' + arg + ')'
    if 'numerictable' in typ_cy:
        return 'make_nt(<PyObject *>' + arg + ')'
    if any(typ_cy.endswith(x) for x in ['__iface__', 'model', '_result']):
        return arg + '.c_ptr if ' + arg + ' != None else <' + typ_cyext + ' *>0'
    if 'std_string' in typ_cy:
        return 'to_std_string(<PyObject *>' + arg + ')'
    return arg

def mk_var(name='', typ='', const='', dflt=None, inpt=False, algo=None):
    '''Return an object with preformatted attributes for given argument.
       Analyses, normalizes and formats types, names and members.
       We can also craete an empty object, which can then be used in jinja2 filters without an extra condition.
       emtpy/optional state is indicated by name==''
    '''
    class fmt_var(object):
        def __init__(self, name, typ, const, dflt, inpt, algo):
            d4pname = ''
            if name:
                value      = name.strip()
                name       = value.rsplit('::', 1)[-1]
                d4pname    = name.replace('lambda', 'lambda_') # we cannot use attr/var with that name, it's a python keyword

                # Now look at the type, first see if it's a ref or ptr
                if const:
                    const  = const.strip() + ' '
                ref        = '&' if '&' in typ else ''
                ptr        = '*' if '*' in typ or any(typ.endswith(x) for x in ['Ptr', '__iface__']) else ''
                # get rid of ref/ptr in type
                realtyp    = typ.replace('&', '').replace('*', '').strip()

                # we try to identify enum types, which become strings in python
                if '::' in realtyp and not any(x in realtyp for x in ['Ptr', 'std::']):
                    # string to enum dict for our algo, needed for converting python strings to C++ enums
                    s2e = 's2e_algorithms_{}'.format(flat(realtyp.rsplit('::', 1)[0]))
                    typ = 'std::string'
                    ref = '&'
                    todaal_member = '({})string2enum(_{}, {})'.format(realtyp, d4pname, s2e)
                else:
                    # any other typ
                    s2e = 's2e_algorithms_{}'.format(algo)
                    typ = realtyp
                    todaal_member = '_{}'.format(d4pname)
                # normalize types from DAAL
                typ_flat = flat(typ)
                typ_cyext = typ_flat
                # interface types are passed py non-const pointer
                # for the cython extern def we use a typedef with prefix 'c_'
                if typ_flat.endswith('__iface__'):
                    typ_cyext  = 'c_'+typ_flat
                    ptr = '*'
                    const = ''
                # for sphinx docu we want to be a bit more readable
                typ_sphinx = typ_flat.replace('std_string', 'str').replace('data_management_numerictable', 'array').lower()
                # in cython/python we want everything to be lower case
                typ_cy = typ_flat.lower()
                # all daal objects are passed as SharedPointer through their *Ptr typedefs
                # we don't want to see the ptr in our call names
                if typ_cy.endswith('ptr'):
                    typ_cy = typ_cy[:-3]
                    const = ''
                # some types we accept without type in cython, because we have custom converters
                notyp_cy = ['data_or_file', 'std_string', 'numerictable']

                # we have a few C (not C++) interfaces, usually not a problem for types
                decl_c = '{}{} {}'.format(typ_flat, ptr, d4pname)
                arg_c = d4pname
                # arrays/tables need special handling for C: they are passed as (ptr, dim1, dim2)
                if typ_cy == 'data_or_file':
                    decl_c = 'double* {0}_ptr, size_t {0}_nrows, size_t {0}_ncols'.format(d4pname)
                    arg_c = 'new data_or_file(daal::data_management::HomogenNumericTable< double >::create({0}_ptr, {0}_ncols, {0}_nrows))'.format(d4pname)
                    const = ''
                # default values (see above pydefaults)
                if dflt != None:
                    pd = (pydefaults[typ] if dflt == True else dflt).rsplit('::', 1)[-1].replace('NumericTablePtr()', 'None')
                    default_val = '"{}"'.format(pd) if typ == 'std::string' else '{}'.format(pd)
                    sphinx_default = default_val
                    pydefault = ' = {}'.format(default_val)
                    cppdefault = ' = {}'.format(cppdefaults[typ] if dflt == True else dflt) if dflt != None else ''
                    if default_val == 'None':
                        default_val = '"None"'
                else:
                    default_val = None
                    pydefault = ''
                    cppdefault = ''
                    sphinx_default = ''
                assert(' ' not in typ), 'Error in parsing variable "{}"'.format(decl)

                hpat_dist = 'REP' if any(x in d4pname for x in ['model', 'inputCentroids']) else 'OneD'

            self.name          = d4pname
            self.daalname      = name
            self.value         = value if name else ''
            self.typ_cpp       = typ if name else ''
            self.arg_cpp       = d4pname
            self.arg_py        = d4pname
            self.arg_cyext     = cy_callext(d4pname, typ_cy, typ_cyext, s2e) if name else ''
            self.arg_c         = arg_c if name else ''
            self.decl_c        = decl_c if name else ''
            self.decl_cyext    = '{}{} {}'.format(typ_cyext, ref if ref != '' else ptr, d4pname) if name else ''
            self.decl_cy       = '{}{}'.format('' if any (x in typ_cy for x in notyp_cy) else typ_cy+' ', d4pname) if name else ''
            self.decl_dflt_cpp = '{}{}{} {}{}'.format(const, typ, ref if ref != '' else ptr, d4pname, cppdefault) if name else ''
            self.decl_dflt_cy  = '{}{}{}'.format('' if any (x in typ_cy for x in notyp_cy) else typ_cy+' ', d4pname, pydefault) if name else ''
            self.decl_cpp      = '{}{}{} {}'.format(const, typ_cyext, ref if ref != '' else ptr, d4pname) if name else ''
            self.decl_member   = '{}{} _{}'.format(typ_cyext, ptr, d4pname) if name else ''
            self.arg_member    = '_{}'.format(d4pname) if name else ''
            self.init_member   = '_{}({})'.format(d4pname, 'NULL' if ptr else '') if inpt else '_{0}({0})'.format(d4pname) if name else ''
            self.assign_member = '_{0} = {0}'.format(d4pname) if name else ''
            self.todaal_member = todaal_member if name else ''
            self.spec          = '("{}", "{}", {}, "{}")'.format(d4pname, typ_cy.replace('data_management_', ''), default_val, hpat_dist) if name else ''
            self.sphinx        = ':param {} {}:{}'.format(typ_sphinx, d4pname, ' [optional, default: {}]'.format(sphinx_default) if sphinx_default else '') if name else ''

        def format(self, s, *args):
            '''Helper function to format a string with attributes from given var
               {}s are replaced with the respective attributes given in args
            '''
            a = [getattr(self, x) for x in args]
            return s.format(*a) if self.name else ''

    return fmt_var(name, typ, const, dflt, inpt, algo)
