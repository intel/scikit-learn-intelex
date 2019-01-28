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
###############################################################################
# Logic to parse DAAL's C++ headers and extract the information for generating
# a config used for generating SWIG interface files.
#
# The parsing is almost context-free and staggering simple. It is by no means
# intended to parse C++ in any other context.
#
# We define several small parsers, each covering a special purpose (extracting a specific feature).
# Each parser accepts line by line, parses it and potentially fills the globl dict.
# When returning False another parser might be applied, if returning True the
# entire line was consumed and there is nothing more to extract.
# The global dict is the "gdict" attribute of the context argument.
# It maps dict-names to dicts. Each such dict contains the extracted data of interesting objects.
#  - namespace:                                         gdict['ns'])
#  - #include files:                                    gdict['includes'])
#  - classes/structs (even if non-template):            gdict['classes']
#    - list of parent classes:                          gdict['classes'][class].parent
#    - public class members:                            gdict['classes'][class].members
#    - get/set methods (formatted for SWIG %rename):    gdict['classes'][class].setgets
#    - template member functions:                       gdict['classes'][class].templates
#    - result type of Batch, Online, distributed:       gdict['classes'][class].result
#  - enums:                                             gdict['enums']
#  - typedefs:                                          gdict['typedefs']
#  - at least one class/func uses a compute method:     gdict['needs_methods']
#  - string of errors encountered (templates)           gdict['error_template_string']
#  - set of steps found (for distributed)               gdict['steps']
# Note that (partial) template class specializations will get separate entries in 'classes'.
# The specializing template arguments will get appended to the class name and the member
# attribute 'partial' set to true.
#
# The context keeps parsing context/state such as the current class. Parser specific
# states should be stored in the parser object itself.
# The context also holds a list of classes/functions which should be ignored.
#
# Expected C++ format/layout
#   - every file defines at most one linear namespace hierachy
#     an error will occur if there is more than one namespace in another
#   - namespace declarations followed by '{' will be ignored
#     - only to be used for forward declarations which will be ignored
#   - the innermost namespace must be 'interface1'
#   - enum values are defined one per separate line
#   - templates are particularly difficult
#     - "template<.*>" should be on a separate line than template name
#     - "template<.*>" must be on a single line
#     - implementation body should start on a separate line
#     - special types we detect:
#       - steps
#       - methods: the argument type must end with 'Method'
#         we do not understand or map the actual type
#   - end of struct/class/enum/namespace '};' must be in a separate line
#     any other code in the same line will result in errorneous parsing
#     leading spaces are not allowed!
#   - within a class-declaration, '}\s*;' is not allowed
#     (except to end the class-def)
#   - forward declarations for non-template classes/structs are ignored
#     (such as "class myclass;")
#   - template member functions for getting/setting values are only supported
#     for a single template argument (fptype) and no parameters
###############################################################################
###############################################################################

from collections import defaultdict, OrderedDict
import re

###############################################################################
class cpp_class(object):
    """A C++ class representation"""
    def __init__(self, n, tpl, parent=None, partial=False):
        self.name = n             # name of the class
        self.template_args = tpl  # list of template arguments as [name, values, default]
        self.members = OrderedDict() # dictionary mapping member names to their types
        self.sets = OrderedDict() # dictionary mapping set enum type to input type
        self.arg_sets = OrderedDict() # dictionary mapping set enum type to input type and argument
        self.arg_gets = OrderedDict() # dictionary mapping set enum type to input type and argument
        self.setgets = []         # set and get methods, formatted for %rename
        self.gets = {}            # getXXX methods and return type
        self.templates = []       # template member functions
        self.partial = partial    # True if this represents a (partial) spezialization
        self.parent = parent      # list of parent classes
        self.result = None        # Result type (if Batch, Online or Disributed)
        self.typedefs = {}        # Typedefs
        assert not partial or tpl

###############################################################################


###############################################################################
class ns_parser(object):
    """parse namespace declaration"""
    def parse(self, l, ctxt):
        m = re.match(r'namespace +(\w+)(.*)', l)
        if m and (not m.group(2) or '{' not in m.group(2)):
            ctxt.gdict['ns'].append(m.group(1))
            return True
        return False


###############################################################################
class eos_parser(object):
    """detect end of struct/class/enum '};'"""
    def parse(self, l, ctxt):
        m = re.match(r'^}\s*;\s*$', l)
        if m:
            ctxt.enum = False
            ctxt.curr_class = False
            ctxt.access = False
            ctxt.template = False
            return True
        return False


###############################################################################
class include_parser(object):
    """parse #include"""
    def parse(self, l, ctxt):
        mi = re.match(r'#include\s+[<\"](algorithms/.+?h)[>\"]', l)
        if mi:
            ctxt.gdict['includes'].add(mi.group(1))
            return True
        return False


###############################################################################
class typedef_parser(object):
    """Parse a typedef"""
    def parse(self, l, ctxt):
        m =  re.match(r'\s*typedef(\s+(struct|typename))?\s+(.+)\s+(\w+).*', l)
        if m:
            if ctxt.curr_class:
                ctxt.gdict['classes'][ctxt.curr_class].typedefs[m.group(4).strip()] = m.group(3).strip()
            else:
                ctxt.gdict['typedefs'][m.group(4).strip()] = m.group(3).strip()
            return True
        return False


###############################################################################
class enum_parser(object):
    """Parse an enum"""
    def parse(self, l, ctxt):
        me = re.match(r'\s*enum +(\w+)\s*', l)
        if me:
            ctxt.enum = me.group(1)
            # we need a deterministic order when generating API
            ctxt.gdict['enums'][ctxt.enum] = OrderedDict()
            return True
        # if found enum Method, extract the enum values
        if ctxt.enum:
            me = re.match(r'.*?}.*', l)
            if me:
                ctxt.enum = False
                return True
            else:
                me = re.match(r'^\s*(\w+)(?:\s*=\s*((\(int\))?\w(\w|:|\s|\+)*))?(\s*,)?\s*((/\*|//).*)?$', l)
                if me and not me.group(1).startswith('last'):
                    ctxt.gdict['enums'][ctxt.enum][me.group(1)] = me.group(2) if me.group(2) else ''
                    return True
        return False


###############################################################################
class access_parser(object):
    """Parse access specifiers"""
    def parse(self, l, ctxt):
        if ctxt.curr_class:
            am =  re.match(r'\s*(public|private|protected)\s*:\s*', l)
            if am:
                ctxt.access = am.group(1) == 'public'
        return False


###############################################################################
class step_parser(object):
    """Look for distributed steps"""
    def parse(self, l, ctxt):
        m =  re.match(r'.*[<, ](step\d+(Master|Local))[>, ].*', l)
        if m:
            ctxt.gdict['steps'].add(m.group(1))
        return False


###############################################################################
class setget_parser(object):
    """Parse a set/get methods"""
    def parse(self, l, ctxt):
        if ctxt.curr_class and ctxt.access:
            mgs = re.match(r'\s*using .+::(get|set);', l)
            if mgs:
                assert not ctxt.template
                ctxt.gdict['classes'][ctxt.curr_class].setgets.append(l.strip(' ;'))
                return True
            mgs = re.match(r'(\s*)([^\(=\s]+\s+)((get|set)(\(((\w|:)+).*\)))', l)
            if mgs:
                assert not ctxt.template
                ctxt.gdict['classes'][ctxt.curr_class].setgets.append([mgs.group(4), mgs.group(2), mgs.group(6), mgs.group(3)])
                # map input-enum to object-type and optional arg
                if mgs.group(4) == 'get': # a getter
                    name = mgs.group(6).strip()
                    if(',' in mgs.group(3)):
                        arg = mgs.group(3).strip(')').split(',')[1].strip().split(' ')
                        ctxt.gdict['classes'][ctxt.curr_class].arg_gets[name] = (mgs.group(2).strip(), arg)
                    else:
                        ctxt.gdict['classes'][ctxt.curr_class].gets[name] = mgs.group(2).strip()
                else: # a setter
                    args = mgs.group(3).split('{')[0].strip(')').strip().split(',')
                    name = mgs.group(6).strip()
                    typ  = args[-1].replace('const', '').strip().split(' ')[0].strip()
                    if len(args) > 2:
                        val  = args[1].strip().split(' ')
                        ctxt.gdict['classes'][ctxt.curr_class].arg_sets[name] = (typ, val)
                    else:
                        ctxt.gdict['classes'][ctxt.curr_class].sets[name] = typ
                return True
            mgs = re.match(r'\s*(?:(?:virtual|DAAL_EXPORT)\s*)?((\w|:|<|>)+)([*&]\s+|\s+[&*]|\s+)(get\w+)\(\s*\)', l)
            if mgs:
                name = mgs.group(4)
                if name not in ['getSerializationTag']:
                    if ctxt.template and name.startswith('get'):
                        assert len(ctxt.template) == 1 and ctxt.template[0][1] == 'fptypes'
                        ctxt.gdict['classes'][ctxt.curr_class].gets[name] = ('double', '<double>')
                        return False
                    else:
                        ctxt.gdict['classes'][ctxt.curr_class].gets[name] = mgs.group(1)
                        return True
            # some get-methods accept an argument!
            # We support only a single argument for now, and only simple types like int, size_t etc, no refs, no pointers
            mgs = re.match(r'\s*(?:virtual\s*)?((\w|:|<|>)+)([*&]\s+|\s+[&*]|\s+)(get\w+)\(\s*((?:\w|_)+)\s+((?:\w|_)+)\s*\)', l)
            if mgs and mgs.group(4) not in ['getResult', 'getInput']:
                ctxt.gdict['classes'][ctxt.curr_class].gets[mgs.group(4)] = (mgs.group(1), mgs.group(5), mgs.group(6))
                return True
        return False


###############################################################################
class result_parser(object):
    """Look for result type"""
    def parse(self, l, ctxt):
        if ctxt.curr_class and ctxt.access and any(x in ctxt.curr_class for x in ['Batch', 'Online', 'Distributed']):
            m =  re.match(r'\s*(?:(?:virtual|const)\s+)?((\w|::|< *| *>)+)\s+(getResult\w*).+', l)
            if m and not m.group(3).endswith('Impl') and not 'return' in m.group(1):
                ctxt.gdict['classes'][ctxt.curr_class].result = (m.group(1), m.group(3))
        return False


###############################################################################
class member_parser(object):
    """Parse class members"""
    def parse(self, l, ctxt):
        if ctxt.curr_class and ctxt.access:
            mm = re.match(r'\s*((?:[\w:_]|< ?| ?>| ?, ?)+)(?<!return|delete)\s+([\w_]+)\s*;', l)
            if mm :
                if mm.group(2) not in ctxt.gdict['classes'][ctxt.curr_class].members:
                    ctxt.gdict['classes'][ctxt.curr_class].members[mm.group(2)] = mm.group(1)
                return True
        return False


###############################################################################
class class_template_parser(object):
    """Parse a template statement"""
    def parse(self, l, ctxt):
        # not a namespace, no enum Method, let's see if it's a template statement
        # this is checking if we have explicit template instantiation here, which we will simply ignore
        mt = re.match(r'\s*template\s+(class|struct)\s+([\w_]+\s*)+(<[\w_ ,:]+>);', l)
        if mt:
            return True
        # this is checking if we have a template specialization here, which we will simply ignore
        mt = re.match(r'\s*template<>\s*(?!(class|struct))[\w_]+.*', l)
        if mt:
            return True
        # now we test for a "proper" template declaration
        mt = re.match(r'\s*template\s*(<.*?>)', l)
        if mt:
            ctxt.template = mt.group(1)
            # we do not reset ctxt.template unless we mapped it to a class/function
            #    or the next line is nothing we can digest
            # we now do some formatting of common template parameter lists
            tmp = ctxt.template.split(',')
            tmplargs = []
            for ta in tmp:
                tmpltmp = None
                mtm = re.match(r'.*?(\w*Method) +(\w+?)( *= *(\w+))?[ >]*$', ta)
                if mtm and not 'CompressionMethod' in l:
                    tmpltmp = [mtm.group(2), mtm.group(1), mtm.group(4) if mtm.group(4) else '']
                    ctxt.gdict['need_methods'] = True
                else:
                    mtt = re.match(r'.*typename \w*?FPType( *= *(\w+))?[ >]*$', ta)
                    if mtt:
                        tmpltmp = ['fptype', 'fptypes', mtt.group(2) if mtt.group(2) else '']
                    else:
                        mtt = re.match(r'.*ComputeStep \w+?( *= *(\w+))?[ >]*$', ta)
                        if mtt:
                            tmpltmp = ['step', 'steps', mtt.group(2) if mtt.group(2) else '']
                if not tmpltmp:
                    tatmp = ta.split('=')
                    tatmp2 = tatmp[0].split()
                    if len(tatmp2) > 1:
                        tmpltmp = [tatmp2[1].strip('<> '), tatmp2[0].strip('<> '), tatmp[-1].strip('<> ') if len(tatmp) > 1 else '']
                    else:
                        tmpltmp = [ta, '', '']
                tmplargs.append(tmpltmp)
            ctxt.template = tmplargs
        # we don't have a 'else' (e.g. if not a template) here since we could have template one-liners
        # is it a class/struct?
        m = re.match(r'(?:^\s*|.*?\s+)(class|struct)\s+(DAAL_EXPORT\s+)?(\w+)\s*(<[^>]+>)?(\s*:\s*((public|private|protected)\s+(.*)))?({|$|:|;)', l)
        m2 = re.match(r'\s*(class|struct)\s+\w+;', l) # forward deckarations can be ignored
        if m and not m2:
            if m.group(3) in ctxt.ignores:
                pass
                # error_template_string += fname + ':\n\tignoring ' + m.group(3)
            else:
                # struct/class
                ctxt.curr_class = m.group(3)
                parents = m.group(8).split(',') if  m.group(8) else []
                parents = [x.replace('public', '').strip(' {};') for x in parents]
                if m.group(4):
                    # template specialization
                    targs = m.group(4).split(',')
                    targs = [a.strip('<> ') for a in targs if not any(ta[0] in a.lower() for ta in ctxt.template)]
                    ctxt.curr_class += '<' + ', '.join(targs) + '>'
                    # print('Found specialization ' + ctxt.curr_class + '\n  File "' + ctxt.header+'", line '+str(ctxt.n))
                    ctxt.gdict['classes'][ctxt.curr_class] = cpp_class(ctxt.curr_class, ctxt.template, parent=parents, partial=True)
                else:
                    ctxt.gdict['classes'][ctxt.curr_class] = cpp_class(ctxt.curr_class, ctxt.template, parent=parents)
                #elif ctxt.template:
                #        ctxt.gdict['error_template_string'] += '$FNAME:' + str(ctxt.n) + ': Warning: Expected a template specialization for class ' + ctxt.curr_class + '\n'
                if ctxt.template:
                    ctxt.template = False
                ctxt.access = (m.group(1) != 'class')
        elif ctxt.template:
            # we only look for member functions if it's a template
            m = re.match(r'\s*((static|const|inline|DAAL_EXPORT)\s+)*(([:\*&\w<>]| >)+\s+)?[&\*]?(\w+)\s*\(.*', l)
            if m and ctxt.access:
                if m.group(5) not in ctxt.ignores:
                    ctxt.gdict['classes'][ctxt.curr_class].templates.append([ctxt.curr_class + '::' + m.group(5), ctxt.template])
                    ctxt.template = False
                    return True
                else:
                    pass
                #error_template_string += fname + ':\n\tignoring ' + m.group(5)
            elif ctxt.access and not mt and not m and not any(s in l for s in ctxt.ignores):
                # not a class but a non-mapped template
                ctxt.gdict['error_template_string'] += '$FNAME:' + str(ctxt.n) + ': Warning:\n\t' + ctxt.template + '\n\t' + l
        # the else case means we have a template-statement, class, method to follow next line
        if not mt:
            # let's keep track of occurences of 'template' which we could not digest
            ctxt.template = False
            mt = re.match(r'template[^<]*<', l)
            if mt:
                ctxt.gdict['error_template_string'] += '$FNAME:' + str(ctxt.n) + ': Warning: ' + l
        return False

###############################################################################
class pcontext(object):
    """Parsing context to keep state between lines"""
    def __init__(self, gdict, ignores, header):
        self.gdict = gdict
        self.ignores = ignores
        self.enum = False
        self.curr_class = False
        self.template = False
        self.access = False
        self.header = header


###############################################################################
# parse a DAAL header file and extract information relevant for SWIG interface files
# Common template argument lists are formatted properly for use
#   in interface files config dict
# Also returns string for errors in parsing templates
def parse_header(header, ignores):
    gdict = defaultdict(list)
    gdict.update( { 'ns': [],
                    'classes': {},
                    'includes': set(),
                    'steps': set(),
                    'need_methods': False,
                    'error_template_string': '',
                    'enums': defaultdict(lambda: defaultdict(lambda: '')),
                    'typedefs': {},
                })
    ctxt = pcontext(gdict, ignores, header.name)
    parsers = [ns_parser(), include_parser(), eos_parser(), typedef_parser(), enum_parser(), access_parser(), step_parser(), setget_parser(), result_parser(), member_parser(), class_template_parser()]

    # go line by line
    ctxt.n = 1
    for l in header:
        # first strip of eol comments
        l = l.split('//')[0]
        # apply each parser, continue to next line if possible
        for p in parsers:
            if p.parse(l, ctxt):
                break
        ctxt.n += 1

    return gdict


def parse_version(header):
    """Parse DAAL version strings"""
    v = (None, None, None)
    for l in header:
        if '#define __INTEL_DAAL_' in l:
            m = re.match(r'#define __INTEL_DAAL__ (\d+)', l)
            if m:
                v = (m.group(1), v[1], v[2])
            m = re.match(r'#define __INTEL_DAAL_MINOR__ (\d+)', l)
            if m:
                v = (v[0], m.group(1), v[2])
            m = re.match(r'#define __INTEL_DAAL_UPDATE__ (\d+)', l)
            if m:
                v = (v[0], v[1], m.group(1))
        if None not in v:
            return v
    return v


if __name__ == "__main__":
    pass
