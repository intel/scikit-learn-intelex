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

###############################################################################
# Top level code for generating code for building daal4py
# - Uses parse.py to read in DAAL C++ headers files.
# - Extracts necessary information like enum values, namespaces, templates etc
# - Uses wrapper_gen.py to generate code
#   - C++ code to construct objects and call algorithms (shared and distributed memory)
#   - Cython code to generate python API
###############################################################################

import glob, os, re
from pprint import pformat, pprint
from os.path import join as jp
from collections import defaultdict, OrderedDict
from jinja2 import Template
from .parse import parse_header
from .wrappers import required, ignore, defaults, specialized, has_dist, ifaces, no_warn, no_constructor, fallbacks, add_setup
from .wrapper_gen import wrapper_gen, typemap_wrapper_template

try:
    basestring
except NameError:
    basestring = str

def cpp2hl(cls):
    return cls.replace('::', '_')

###############################################################################
def cleanup_ns(fname, ns):
    """return a sanitized namespace name"""
    # strip of namespace 'interface1'
    while len(ns) and ns[-1].startswith('interface'):
        del ns[-1]
    # cleanup duplicates
    while len(ns) >= 2 and ns[-1] == ns[len(ns)-2]:
        del ns[-1]
    # we should now have our namespace hierachy
    if len(ns) == 0 or ns[0] != 'daal':
        print(fname + ":0: Warning: No namespace (starting with daal) found in " + fname + '. Ignored.')
        return False
    nsn = '::'.join(ns[1:])
    # namespace 'daal' is special, it's empty
    if len(nsn) == 0 :
        nsn = 'daal'
    # Multiple namespaces will leave 'interface' in the hierachy
    # we cannot handle these cases
    if 'interface' in nsn:
        print(fname + ":0: Warning: Multiple namespaces found in " + fname + '. Ignored.')
        return False
    return nsn


###############################################################################
###############################################################################
def splitns(x):
    '''Split string at last '::' '''
    tmp_ = x.rsplit('::', 1)
    if len(tmp_) == 1:
        return ('', x)
    else:
        return tmp_

def get_parent(ns):
    tmp = ns.rsplit('::', 1)
    return tmp[0] if len(tmp) > 1 else 'daal'

###############################################################################
class namespace(object):
    """Holds all extracted data of a namespace"""
    def __init__(self, name):
        self.classes = {}
        self.enums = {}
        self.typedefs = {}
        self.headers = []
        self.includes = set()
        self.name = name
        self.need_methods = False
        self.steps = set()
        self.children = set()

###############################################################################
def ignored(ns, a=None):
    return ns in ignore and ((a != None and a in ignore[ns]) or (a == None and not ignore[ns]))


###############################################################################
###############################################################################
class cython_interface(object):
    """collecting and generating data for code generation"""

    # classes/functions we generally ignore
    ignores = ['AlgorithmContainerIface', 'AnalysisContainerIface',
               'PredictionContainerIface', 'TrainingContainerIface', 'DistributedPredictionContainerIface',
               'BatchContainerIface', 'OnlineContainerIface', 'DistributedContainerIface',
               'BatchContainer', 'OnlineContainer', 'DistributedContainer',
               'serializeImpl', 'deserializeImpl', 'serialImpl',
               'getEpsilonVal', 'getMinVal', 'getMaxVal', 'getPMMLNumType', 'getInternalNumType', 'getIndexNumType',
               'allocateNumericTableImpl', 'allocateImpl',
               'setPartialResultStorage', 'addPartialResultStorage',]

    # files we ignore/skip
    ignore_files = ['daal_shared_ptr.h', 'daal.h', 'daal_win.h', 'algorithm_base_mode_batch.h',
                    'algorithm_base.h', 'algorithm.h', 'ridge_regression_types.h', 'kdtree_knn_classification_types.h',
                    'multinomial_naive_bayes_types.h', 'daal_kernel_defines.h', 'linear_regression_types.h',
                    'multi_class_classifier_types.h']

    # default values for all types
    # we replace all default values given in DAAL with these.
    # Our generated code will detect these and then let DAAL handle setting defaults
    # Note: we assume default bool is always false.
    defaults = defaultdict(lambda: None)
    defaults.update({'double': 'NaN64',
                     'float': 'NaN32',
                     'int': '-1',
                     'long': '-1',
                     'size_t': '-1',
                     'bool': 'False',
                     #'std::string' : '""',
                     'std::string &' : '""',})

    done = []


###############################################################################
    def __init__(self, include_root):
        self.include_root = include_root
        self.namespace_dict = defaultdict(namespace)


###############################################################################
    def read(self):
        """
        Walk through each directory in the root dir and read in C++ headers.
        Creating a namespace dictionary. Of course, it needs to go through every header file to find out
        what namespace it is affiliated with. Once done, we have a dictionary where the key is the namespace
        and the values are namespace class objects. These objects carry all information as extracted by parse.py.
        """
        for (dirpath, dirnames, filenames) in os.walk(self.include_root):
            for filename in filenames:
                if filename.endswith('.h') and not 'neural_networks' in dirpath and not any(filename.endswith(x) for x in cython_interface.ignore_files):
                    fname = jp(dirpath,filename)
                    #print('reading ' +  fname)
                    with open(fname, "r") as header:
                        parsed_data = parse_header(header, cython_interface.ignores)

                    ns = cleanup_ns(fname, parsed_data['ns'])
                    # Now let's update the namespace; more than one file might constribute to the same ns
                    if ns:
                        if ns not in self.namespace_dict:
                            self.namespace_dict[ns] = namespace(ns)
                        pns = get_parent(ns)
                        if pns not in self.namespace_dict:
                            self.namespace_dict[pns] = namespace(pns)
                        if ns != 'daal':
                            self.namespace_dict[pns].children.add(ns)
                        self.namespace_dict[ns].includes = self.namespace_dict[ns].includes.union(parsed_data['includes'])
                        self.namespace_dict[ns].steps = self.namespace_dict[ns].steps.union(parsed_data['steps'])
                        self.namespace_dict[ns].classes.update(parsed_data['classes'])
                        self.namespace_dict[ns].enums.update(parsed_data['enums'])
                        self.namespace_dict[ns].typedefs.update(parsed_data['typedefs'])
                        self.namespace_dict[ns].headers.append(fname.replace(self.include_root, '').lstrip('/'))
                        if parsed_data['need_methods']:
                            self.namespace_dict[ns].need_methods = True


###############################################################################
# Postprocessing starts here
###############################################################################
    def get_ns(self, ns, c, attrs=['classes', 'enums', 'typedefs']):
        """
        Find class c starting in namespace ns.
        We search all entries given in attrs - not only 'classes'.
        c can have qualified namespaces in its name.
        We go up to all enclosing namespaces of ns until we find c (e.g. current-namespace::c)
        or we reached the global namespace.
        """
        # we need to cut off leading daal::
        if c.startswith('daal::'):
            c = c[6:]
        tmp = c.split('::')
        tmp = splitns(c)
        cns = ('::' + tmp[0]) if len(tmp) == 2 else '::' # the namespace-part of our class
        cname = tmp[-1]                                  # class name (stripped off namespace)
        currns = ns + cns   # current namespace in which we look for c
        done = False
        while currns and not done:
            # if in the outmost level we only have cns, which starts with '::'
            tmpns = currns.strip(':')
            if tmpns in self.namespace_dict and any(cname in getattr(self.namespace_dict[tmpns], a) for a in attrs):
                return tmpns
            if ns == 'daal':
                done = True
            else:
                if currns.startswith('::'):
                    ns = 'daal'
                else:
                    ns = splitns(ns)[0]
                currns = ns + cns
        return None


###############################################################################
    def get_all_attrs(self, ns, cls, attr, ons=None):
        """
        Return an ordered dict, combining the 'attr' dicts of class 'cls' and all its parents.
        Note: this looks for parents of 'cls' not parents of 'ns'!
        """
        if not ons:
            ons = ns
        # we need to cut off leading daal::
        cls = cls.replace('daal::', '')
        if ns not in self.namespace_dict or cls not in self.namespace_dict[ns].classes:
            if ns in self.namespace_dict and '::' not in cls:
                # the class might be in one of the parent ns
                ns = self.get_ns(ns, cls)
                if ns == None:
                    return None
            else:
                return None

        pmembers = OrderedDict()
        # we add ours first. When expanding, duplicates from parents will be ignored.
        tmp = getattr(self.namespace_dict[ns].classes[cls], attr)
        for a in tmp:
            n = a if '::' in a else ns + '::' + a
            if not ignored(ons, n):
                pmembers[n] = tmp[a]
        for parent in self.namespace_dict[ns].classes[cls].parent:
            parentclass = cls
            pns = ns
            # we need to cut off leading daal::
            sanep = parent.split()[-1].replace('daal::', '')
            parentclass = splitns(sanep)[1]
            pns = self.get_ns(pns, sanep)
            if pns != None:
                pms = self.get_all_attrs(pns, parentclass, attr, ons)
                for x in pms:
                    # ignore duplicates from parents
                    if not ignored(ons, x) and not any(x == y for y in pmembers):
                        pmembers[x] = pms[x]
        return pmembers


###############################################################################
    def to_lltype(self, t):
        """
        return low level (C++ type). Usually the same as input.
         Only very specific cases need a conversion.
        """
        if t in ['DAAL_UINT64']:
            return 'ResultToComputeId'
        return t

###############################################################################
    def to_hltype(self, ns, t):
        """
        Return triplet (type, {'stdtype'|'enum'|'class'|'?'}, namespace) to be used in the interface
        for given type 't'.
            'stdtype' means 't' is a standard data type understood by cython and plain C++
            'enum' means 't' is a C/C++ enumeration
            'class' means 't' is a regular C++ class
            '?' means we do not know what 't' is
        For classes, we also add lookups in namespaces that DAAL C++ API finds through "using".
        """
        if t in ['DAAL_UINT64']:
            ### FIXME
            t = 'ResultToComputeId'
        tns, tname = splitns(t)
        if t in ['double', 'float', 'int', 'size_t',]:
            return (t, 'stdtype', '')
        if t in ['bool']:
            return ('bool', 'stdtype', '')
        if t.endswith('ModelPtr'):
            thens = self.get_ns(ns, t, attrs=['typedefs'])
            return ('daal::' + thens + '::ModelPtr', 'class', tns)
        if t in ['data_management::NumericTablePtr',] or t in ifaces.values():
            return ('daal::' + t, 'class', tns)
        if 'Batch' in self.namespace_dict[ns].classes and t in self.namespace_dict[ns].classes['Batch'].typedefs:
            tns, tname = splitns(self.namespace_dict[ns].classes['Batch'].typedefs[t])
            return (self.namespace_dict[ns].classes['Batch'].typedefs[t], 'class', tns)
        tt = re.sub(r'(?<!daal::)services::SharedPtr', r'daal::services::SharedPtr', t)
        tt = re.sub(r'(?<!daal::)algorithms::', r'daal::algorithms::', tt)
        if tt in ifaces.values():
            return (tt, 'class', tns)
        tns = self.get_ns(ns, t)
        if tns:
            tt = tns + '::' + tname
            if tt == t:
                return ('std::string &', 'enum', tns) if tname in self.namespace_dict[tns].enums else (tname, 'class', tns)
            else:
                return self.to_hltype(ns, tt)
        else:
            usings = ['algorithms::optimization_solver']
            if not any(t.startswith(x) for x in usings):
                for nsx in usings:
                    r = self.to_hltype(ns, nsx + '::' + t)
                    if r:
                        return r
        return None if '::' in t else (t, '??', '??')


###############################################################################
    def get_values(self, ns, n):
        """
        Return available values for given enum or special "group".
        """
        if n == 'fptypes':
            return ['double', 'float']
        nns = self.get_ns(ns, n)
        if nns:
            nn = splitns(n)[1]
            if nn in self.namespace_dict[nns].enums:
                return [nns + '::' + x for x in self.namespace_dict[nns].enums[nn]]
            return ['unknown_' + nns + '_class_'+n]
        return ['unknown_'+n]


###############################################################################
    def get_tmplarg(self, ns, n):
        """
        Return template argument specifier: "typename" or actual type.
        """
        if n == 'fptypes':
            return 'typename'
        nns = self.get_ns(ns, n)
        if nns:
            nn = splitns(n)[1]
            if nn in self.namespace_dict[nns].enums:
                return nns + '::' + nn
            return 'unknown_' + nns + '_class_'+n
        return 'unknown_'+n


###############################################################################
    def get_class_for_typedef(self, ns, cls, td):
        """
        Find the Result type for given algorithm in the C++ namespace hierachy.
        Strips off potential SharedPtr def.
        """
        if ns not in self.namespace_dict or cls not in self.namespace_dict[ns].classes or td not in self.namespace_dict[ns].classes[cls].typedefs:
            return None

        res =  self.namespace_dict[ns].classes[cls].typedefs[td]
        tmp = splitns(res)
        ret = None
        if res.endswith('Type'):
            # this is a dirty hack: we assume there are no typedefs *Type outside classes
            assert res.endswith(td)
            ret = self.get_class_for_typedef(self.get_ns(ns, tmp[0]), splitns(tmp[0])[1], tmp[1])
            if not ret and '<' in tmp[0]:
                n = tmp[0].split('<', 1)[0]
                ret = self.get_class_for_typedef(self.get_ns(ns, n), splitns(n)[1], tmp[1])
        else:
            ret = tmp
        if ret and ret[1] not in self.namespace_dict[ret[0]].classes and '<' in ret[1]:
            # Probably a template, sigh
            # For now, let's just cut off the template paramters.
            # Let's hope we don't need anything more sophisticated (like if there are actually specializations...)
            c = ret[1].split('<', 1)[0] if ret else res.split('<')[0]
            n = self.get_ns(ns, c)
            ret = (n, splitns(c)[1]) if n else None
            # ret = (self.get_ns(ns, res.split('<')[0]), tmp[1])
        return ret


###############################################################################
    def get_expand_attrs(self, ns, cls, attr):
        """
        Find enum type for attributes in "attr" dict of class "cls" and returns
        2-tuple of lists of tuples
          [0] list of members it can map to a hltype (namespace, member/value name, type of attr)
          [1] list of members it can not map tp hltype (namespace, attribute)
        Only standard data types and those with typemaps can be mapped to a hltype.
        """
        attrs = self.get_all_attrs(ns, cls, attr)
        explist = []
        ignlist = []
        for i in attrs:
            inp = splitns(i)[1]
            ins = self.get_ns(ns, i)
            assert ins
            assert ins in self.namespace_dict
            assert inp in self.namespace_dict[ins].enums
            hlt = self.to_hltype(ns, attrs[i])
            if ignored(ns, '::'.join([ins, inp])):
                continue
            if hlt:
                if hlt[1] in ['stdtype', 'enum', 'class']:
                    for e in self.namespace_dict[ins].enums[inp]:
                        if not any(e in x for x in explist) and not ignored(ins, e):
                            explist.append((ins, e, hlt[0]))
                else:
                    print("// Warning: ignoring " + ns + " " + str(hlt))
                    ignlist.append((ins, i))
            else:
                print("// Warning: could not find hlt for " + ns + ' ' + cls + " " + i + " " + attrs[i])
        return (explist, ignlist)


###############################################################################
    def prepare_resultmaps(self, ns):
        """
        Prepare info about typedefs for Result type of given namespace.
        Uses target language-specific defines/functions
          - native_type: returns native representation of its argument
          - TMGC(n): deals with GC(refcounting for given number of references (R)
        Looks up return type and then target-language independently creates lists of its content.
        """
        jparams = {}
        res = self.get_class_for_typedef(ns, 'Batch', 'ResultType')
        if not res and 'Result' in self.namespace_dict[ns].classes:
            res = (ns, 'Result')
        if res and '_'.join(res) not in self.done:
            self.done.append('_'.join(res))
            attrs = self.get_expand_attrs(res[0], res[1], 'sets')
            if attrs and attrs[0]:
                jparams = {'class_type': 'daal::' + res[0] + '::' + res[1] + 'Ptr',
                           'enum_gets': attrs[0],
                           'named_gets': [],
                       }
            else:
                print('// Warning: could not determine Result attributes for ' + ns)
        elif res:
            jparams = {'class_type': 'daal::' + res[0] + '::' + res[1] + 'Ptr',}
        elif ns not in no_warn or 'Result' not in no_warn[ns]:
            print('// Warning: no result found for ' + ns)
        return jparams


###############################################################################
    def prepare_modelmaps(self, ns, mname='Model'):
        """
        Return string from typemap_wrapper_template for given Model.
        uses entries from 'gets' in Model class def to fill 'named_gets'.
        """
        jparams = {}
        if mname in self.namespace_dict[ns].classes:
            model = self.namespace_dict[ns].classes[mname]
            jparams = {'class_type': 'daal::' + ns + '::ModelPtr',
                       'enum_gets': [],
                       'named_gets': [],
                   }
            huhu = self.get_all_attrs(ns, mname, 'gets')
            for g in huhu:
                if not any(g.endswith(x) for x in ['SerializationTag',]):
                    gn = splitns(g)[1].replace('get', '')
                    if not any(gn == x[1] for x in jparams['named_gets']):
                        jparams['named_gets'].append((huhu[g], gn))
        return jparams


###############################################################################
    def expand_typedefs(self, ns):
        """
        We expand all typedefs in classes/namespaces without recursing
        to outer scopes or namespaces.
        """
        def expand_td(typedefs):
            done = 1
            while done != 0:
                done = 0
                for td1 in typedefs:
                    for td2 in typedefs:
                        if td1 != td2 and (td1+'::') in typedefs[td2]:
                            typedefs[td2] = typedefs[td2].replace(td1, typedefs[td1])
                            done += 1

        expand_td(self.namespace_dict[ns].typedefs)
        for c in self.namespace_dict[ns].classes:
            expand_td(self.namespace_dict[ns].classes[c].typedefs)


###############################################################################
    def order_iargs(self, tmp_input_args, tmp_iargs_decl, tmp_iargs_call):
        """
        We have to put the intput args into the "right" order.
        , e.g. start with data then model, then whatever else
        """
        ordered = ['data', 'model', 'labels', 'dependentVariable', 'dependentVariables',
                   'tableToFill', 'dataForPruning', 'dependentVariablesForPruning', 'labelsForPruning']
        input_args, iargs_decl, iargs_call = [], [], []
        for arg in ordered:
            for i in range(len(tmp_input_args)):
                if tmp_input_args[i][1] == arg:
                    input_args.append(tmp_input_args[i])
                    iargs_decl.append(tmp_iargs_decl[i])
                    iargs_call.append(tmp_iargs_call[i])
        for i in range(len(tmp_input_args)):
            if tmp_input_args[i][1] not in ordered:
                input_args.append(tmp_input_args[i])
                iargs_decl.append(tmp_iargs_decl[i])
                iargs_call.append(tmp_iargs_call[i])
        return (input_args, iargs_decl, iargs_call)


###############################################################################
    def prepare_hlwrapper(self, ns, mode, func):
        """
        Prepare data structures for generating high level wrappers.

        This is main function for generating high level wrappers.
        Here we prepare and return the data structure from which wrappers can be generated.
        The information is extracted from self.namespace_dict.

        We first extract template parameters and setup a generic structure/array. The generated array
        holds the general template spec in entry [0] and specializations in the following entires (if exist).

        We then extract input arguments. The resulting arrays begin with required inputs followed by optional inputs.
        We get the input arguments by expanding the enums from the set methods in the Input class (InputType).
        Each value in the enum becomes a separate input arguments.

        Next we extract parameters - required and optional separately. Each member of parameter_type becomes
        a separate arguments. Default values are handled in the lower C++ level. We provide "generic" default
        values like -1, NULL and "" for *all* parameters. These values cannot be reasonably used by the user.

        Next we extract type-map (native_type) information for Model and Result types.
        """
        if mode in self.namespace_dict[ns].classes:
            ins = splitns(self.namespace_dict[ns].classes[mode].typedefs['InputType'])[0] if 'InputType' in self.namespace_dict[ns].classes[mode].typedefs else ns
            jparams = {'ns': ns,
                       'algo': func,
                       'template_decl': OrderedDict(),
                       'template_spec': [],
                       'template_args': [],
                       'params_opt': OrderedDict(),
                       'params_req': OrderedDict(),
                       's1': 'step1Local',
                       's2': 'step2Master',
                   }
            # at this point required parameters need to be explicitly/maually provided in wrappers.required
            if ns in required and mode in required[ns]:
                jparams['params_req'] = OrderedDict(required[ns][mode])
            f = ''
            tdecl = []
            if self.namespace_dict[ns].classes[mode].template_args:
                if ns in specialized and mode in specialized[ns]:
                    # there might be specialized template classes, we provide explicit specs
                    v = specialized[ns][mode]
                    tdecl.append({'template_decl': v['tmpl_decl'],
                                  'template_args': None,
                                  'pargs': None})
                    for s in v['specs']:
                        tdecl.append({'template_decl': OrderedDict([(x, v['tmpl_decl'][x]) for x in s['template_decl']]),
                                      'template_args': [s['expl'][x] if x in s['expl'] else x for x in v['tmpl_decl']],
                                      'pargs': [s['expl'][x] for x in s['expl']]})
                else:
                    # 'normal' template
                    tdecl.append({'template_decl': OrderedDict([(t[0], {'template_decl': self.get_tmplarg(ns, t[1]),
                                                                        'values': self.get_values(ns, t[1]),
                                                                        'default': t[2].replace('DAAL_ALGORITHM_FP_TYPE', 'double')}) for t in self.namespace_dict[ns].classes[mode].template_args]),
                                  'template_args': [t[0] for t in self.namespace_dict[ns].classes[mode].template_args],
                                  'pargs': None})

            # A parameter can be a specialized template. Sigh.
            # we need to provide specialized template classes for them.
            # We do this by creating a list for templates, for specializations the list-length is >1
            # and the first entry is the base/"real" template spec, following entries are specializations.
            # At some point we might have other things that influence this (like result or input).
            # for now, we only check for Parameter specializations
            decl_opt, decl_req , call_opt, call_req = [], [], [], []
            for td in tdecl:
                if 'template_args' in td:
                    parms = None
                    # this is a "real" template for which we need a body
                    td['params_req'] = OrderedDict()
                    td['params_opt'] = OrderedDict()
                    td['params_get'] = 'parameter'
                    pargs_exp = '<' + ','.join([splitns(x)[1] for x in td['pargs']]) + '>' if td['pargs'] else ''
                    cls = mode + pargs_exp
                    if 'ParameterType' in self.namespace_dict[ns].classes[cls].typedefs:
                        p = self.get_class_for_typedef(ns, cls, 'ParameterType')
                        parms = self.get_all_attrs(p[0], p[1], 'members', ns) if p else None
                        if not parms:
                            if ns in fallbacks and 'ParameterType' in fallbacks[ns]:
                                parms = self.get_all_attrs(*(splitns(fallbacks[ns]['ParameterType']) + ['members', ns]))
                        tmp = '::'.join([ns, mode])
                        if not parms:
                            tmp = '::'.join([ns, mode])
                            if tmp not in no_warn or 'ParameterType' not in no_warn[tmp]:
                                print('// Warning: no members of "ParameterType" found for ' + tmp)
                    else:
                        tmp = '::'.join([ns, mode])
                        if tmp not in no_warn or 'ParameterType' not in no_warn[tmp]:
                            print('// Warning: no "ParameterType" defined for ' + tmp)
                        parms = None
                    if parms:
                        p = self.get_all_attrs(ns, cls, 'members')
                        if not p or not any(x.endswith('parameter') for x in p):
                            td['params_get'] = 'parameter()'
                    else:
                        td['params_get'] = None
                        continue
                    # now we have a dict with all members of our parameter: params
                    # we need to inspect one by one
                    hlts = {}
                    jparams['params_opt'] = OrderedDict()
                    for p in parms:
                        pns, tmp = splitns(p)
                        if not tmp.startswith('_') and not ignored(pns, tmp):
                            hlt = self.to_hltype(pns, parms[p])
                            if hlt and hlt[1] in ['stdtype', 'enum', 'class']:
                                (hlt, hlt_type, hlt_ns) = hlt
                                llt = self.to_lltype(parms[p])
                                needed = True
                                pval = None
                                if hlt_type == 'enum':
                                    pval = '(' + hlt_ns + '::' + llt + ')string2enum(' + tmp + ', s2e_' + hlt_ns.replace(':', '_') + ')'
                                else:
                                    pval = tmp
                                if pval != None:
                                    thetype = (hlt if hlt else parms[p])
                                    if tmp in jparams['params_req']:
                                        td['params_req'][tmp] = pval
                                        decl_req.append('const ' + thetype + ' ' + tmp)
                                        call_req.append(tmp)
                                    else:
                                        td['params_opt'][tmp] = pval
                                        prm = tmp
                                        dflt = defaults[pns][prm] if pns in defaults and prm in defaults[pns] else self.defaults[thetype]
                                        decl_opt.append(' '.join(['const', thetype, prm, '=', str(dflt)]))
                                        call_opt.append(prm)
                                else:
                                    print('// Warning: do not know what to do with ' + pns + ' : ' + p + '(' + parms[p] + ')')
                            else:
                                print('// Warning: parameter member ' + p + ' of ' + pns + ' is no stdtype, no enum and not a DAAl class. Ignored.')

            # endfor td in tdecl

            # Now let's get the input arguments (provided to input class/object of algos)
            tmp_iargs_decl = []
            tmp_iargs_call = []
            tmp_input_args = []
            setinputs = ''
            inp = self.get_class_for_typedef(ns, 'Batch', 'InputType')
            if not inp and 'Input' in self.namespace_dict[ns].classes:
                inp = (ns, 'Input')
            if inp:
                expinputs = self.get_expand_attrs(inp[0], inp[1], 'sets')
                reqi = 0
                for ins, iname, itype in expinputs[0]:
                    tmpi = iname
                    if tmpi and not ignored(ns, tmpi):
                        if ns in defaults and tmpi in defaults[ns]:
                            i = len(tmp_iargs_decl)
                            dflt = ' = ' + defaults[ns][tmpi]
                        else:
                            i = reqi
                            reqi += 1
                            dflt = ''
                        if 'NumericTablePtr' in itype:
                            #ns in has_dist and iname in has_dist[ns]['step_specs'][0].inputnames or iname in ['data', 'labels', 'dependentVariable', 'tableToFill']:
                            itype = 'TableOrFList *'
                        tmp_iargs_decl.insert(i, 'const ' + itype + ' ' + iname + dflt)
                        tmp_iargs_call.insert(i, iname)
                        tmp_input_args.insert(i, (ins + '::' + iname, iname, itype))
            else:
                print('// Warning: no input type found for ' + ns)

            # We have to bring the input args into the "right" order
            jparams['input_args'], jparams['iargs_decl'], jparams['iargs_call'] = self.order_iargs(tmp_input_args, tmp_iargs_decl, tmp_iargs_call)
            jparams['decl_req'] = decl_req
            jparams['call_req'] = call_req
            jparams['decl_opt'] = decl_opt
            jparams['call_opt'] = call_opt
            # we will need something more sophisticated if the interesting parent class is not a direct parent (a grand-parent for example)
            prnts = list(set([cpp2hl(x) for x in ifaces if any(x.endswith(y) for y in self.namespace_dict[ns].classes[mode].parent)]))
            jparams['iface'] = prnts if len(prnts) else list([None])
        else:
            jparams = {}
            tdecl = []
        # here we know parameters, inputs etc for each
        # let's store this
        retjp = {
            'params': jparams,
            'sparams': tdecl,
            'model_typemap': self.prepare_modelmaps(ns),
            'result_typemap': self.prepare_resultmaps(ns),
            'create': '::create' if '::'.join([ns, mode]) in no_constructor else '',
            'add_setup': True if ns in add_setup else False,
        }
        if ns in has_dist:
            retjp['dist'] = has_dist[ns]

        return {ns + '::' + mode : retjp}


    def hlapi(self, algo_patterns):
        """
        Generate high level wrappers for namespaces listed in algo_patterns (or all).

        First extract the namespaces we really want, e.g. ignore NN.

        Then we expand typedefs on class and namespace levels.

        Then generate maps for each algo mapping string arguments to C++ enum values.

        Next prepares parsed data for code generation (e.g. setting up dicts for jinja).

        Finally generates
          - Result/Model classes and their properties
          - algo-wrappers
          - initialization

        We generate strings for a C++ header, a C++ file and a cython file.
        """
        tmaps, wrappers, hlapi, dtypes = '', '', '', ''
        algoconfig = {}

        algos = [x for x in self.namespace_dict if any(y in x for y in algo_patterns)] if algo_patterns else self.namespace_dict
        algos = [x for x in algos if not any(y in x for y in ['quality_metric', 'transform'])]

        # First expand typedefs
        for ns in algos + ['algorithms::classifier', 'algorithms::linear_model',]:
            self.expand_typedefs(ns)
        # Next, extract and prepare the data (input, parameters, results, template spec)
        for ns in algos + ['algorithms::classifier', 'algorithms::linear_model',]:
            if not ignored(ns):
                nn = ns.split('::')
                if nn[0] == 'daal':
                    if nn[1] == 'algorithms':
                        func = '_'.join(nn[2:])
                    else:
                        func = '_'.join(nn[1:])
                elif nn[0] == 'algorithms':
                    func = '_'.join(nn[1:])
                else:
                    func = '_'.join(nn)
                algoconfig.update(self.prepare_hlwrapper(ns, 'Batch', func))

        # and now we can finally generate the code
        wg = wrapper_gen(algoconfig, {cpp2hl(i): ifaces[i] for i in ifaces})
        cpp_map, cpp_begin, cpp_end, pyx_map, pyx_begin, pyx_end = '', '', '#define NO_IMPORT_ARRAY\n#include "daal4py_cpp.h"\n', '', '', ''

        for ns in algos:
            if ns.startswith('algorithms::') and not ns.startswith('algorithms::neural_networks') and self.namespace_dict[ns].enums:
                cpp_begin += 'static std::map< std::string, int64_t > s2e_' + ns.replace(':', '_') + ' =\n{\n'
                for e in  self.namespace_dict[ns].enums:
                    for v in self.namespace_dict[ns].enums[e]:
                        vv = ns + '::' + v
                        cpp_begin += ' '*4 +'{"' + v + '", ' + vv + '},\n'
                cpp_begin += '};\n\n'

        hlargs = {}
        for a in algoconfig:
            (ns, algo) = splitns(a)
            if algo.startswith('Batch'):
                tmp = wg.gen_wrapper(ns, algo)
                if tmp:
                    cpp_map   += tmp[0]
                    cpp_begin += tmp[1]
                    cpp_end   += tmp[2]
                    pyx_map   += tmp[3]
                    pyx_begin += tmp[4]
                    pyx_end   += tmp[5]
                    dtypes    += tmp[6]
                    hlargs[ns] = tmp[7]

        hds = wg.gen_headers()
        fts = wg.gen_footers(dtypes)

        pyx_end += fts[1]
        # we add a comment with tables providing parameters for each algorithm
        # might be useful for generating docu
        cpp_end += fts[0] + '\n/*\n'
        for algo in hlargs:
            if len(hlargs[algo]):
                cpp_end += 'Algorithm:' + cpp2hl(algo.replace('algorithms::', '')) + '\n'
                cpp_end += 'Name,Type,Default\n'
                for a in hlargs[algo]:
                    cpp_end += ','.join([str(x).rsplit('::')[-1] for x in a]).replace('const', '').replace('&', '').strip() + '\n'
                cpp_end += '\n'
        cpp_end += '\n*/\n'
        # Finally combine the different sections and return the 3 strings
        return(hds[0] + cpp_map + cpp_begin + '\n#endif', cpp_end, hds[1] + pyx_map + pyx_begin + pyx_end)


###############################################################################
###############################################################################
###############################################################################
###############################################################################

def gen_daal4py(daalroot, outdir, warn_all=False):
    global no_warn
    if warn_all:
        no_warn = {}

    iface = cython_interface(jp(daalroot, 'include', 'algorithms'))
    iface.read()
    cpp_h, cpp_cpp, pyx_file = iface.hlapi(['kmeans',
                                            'pca',
                                            'svd',
                                            'multinomial_naive_bayes',
                                            'linear_regression',
                                            'multivariate_outlier_detection',
                                            'univariate_outlier_detection',
                                            'svm',
                                            'kernel_function',
                                            'multi_class_classifier',
                                            'gbt',
                                            'engine',
                                            'decision_tree',
                                            'decision_forest',
                                            'ridge_regression',
                                            'optimization_solver',
                                            'logistic_regression',
    ])
    # 'ridge_regression', parametertype is a template without any need
    with open(jp(outdir, 'daal4py_cpp.h'), 'w') as f:
        f.write(cpp_h)
    with open(jp(outdir, 'daal4py_cpp.cpp'), 'w') as f:
        f.write(cpp_cpp)
    with open(jp(outdir, 'daal4py_cy.pyx'), 'w') as f:
        f.write(pyx_file)
