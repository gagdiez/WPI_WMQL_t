import ast
from os import path
from copy import deepcopy
from operator import lt, gt
from itertools import takewhile
import fnmatch

from .code_util import DocStringInheritor


__all__ = [
    'keywords', 'EvaluateQueries', 'eval_queries', 'eval_queries_volume',
    'queries_syntax_check', 'queries_preprocess',
    'TractQuerierSyntaxError', 'TractQuerierLabelNotFound'
]

keywords = [
    'and',
    'or',
    'not in',
    'not',
    'only',
    'endpoints_in',
    'both_endpoints_in',
    'anterior_of',
    'posterior_of',
    'medial_of',
    'lateral_of',
    'inferior_of',
    'superior_of',
]

class VolumeQueryInfo(object):
    """Information about a processed query on volumes"""

    def __init__(self, inclusions=[], exclusions=[], seeds=[]):
        self.inclusions = inclusions
        self.exclusions = exclusions
        self.seeds = seeds

    def _check_only_one_inclusion(self, qinfo):
        if qinfo.exclusions or qinfo.seeds or len(qinfo.inclusions) > 1:
            raise NotImplementedError("We cannot compute this query yet")

    def union(self, other):
        # This could be changed to include (AnB)uC or Au(BnC)
        # but (AnB)u(CnD) is impossible to do with masks
        inclusions = []
        try:
            self._check_only_one_inclusion(self)
            for other_inclusion in other.inclusions:
                inclusions += [self.inclusions[0] + other_inclusion]
        except:
            self._check_only_one_inclusion(other)
            for self_inclusion in self.inclusions:
                inclusions += [self_inclusion + other.inclusions[0]]
        return VolumeQueryInfo(inclusions)

    def substract(self, other):
        self._check_only_one_inclusion(self)
        self._check_only_one_inclusion(other)

        self_mask = self.inclusions[0]
        other_mask = other.inclusions[0]
        new_mask = self_mask * 0

        nzr = self_mask.nonzero()
        new_mask[nzr] = self_mask[nzr] * ~other_mask[nzr]

        return VolumeQueryInfo([new_mask])

    def intersect_mask(self, other):
        self._check_only_one_inclusion(self)
        self._check_only_one_inclusion(other)

        return VolumeQueryInfo([self.inclusions[0] * other.inclusions[0]])

    def intersection(self, other):
        return VolumeQueryInfo(self.inclusions + other.inclusions,
                               self.exclusions + other.exclusions,
                               self.seeds + other.seeds)

    def copy(self):
        return VolumeQueryInfo(self.inclusions, self.exclusions, self.seeds)

    def to_seed_mask(self):
        import numpy as np
        if self.exclusions or self.seeds:
            raise ValueError("We cannot transform a seed mask or exclusion "
                             "mask in end points")

        if len(self.inclusions) > 1:
            # We have an intersection, therefore, we want the endpoints to be
            # were all the masks intersect
            self.inclusions = [np.prod(self.inclusions, axis=0)]

        self.seeds = list(self.inclusions)
        self.inclusions = []
        return self

    def negate(self):
        for i in range(len(self.inclusions)):
            self.inclusions[i] = ~self.inclusions[i]

        for i in range(len(self.seeds)):
            self.seeds[i] = ~self.seeds[i]

        for i in range(len(self.exclusions)):
            self.exclusions[i] = ~self.exclusions[i]

        return self

    def exclude(self):
        if self.exclusions:
            raise ValueError("We cannot exclude conjunctions")

        self.exclusions = self.inclusions
        self.exclusions += self.seeds
        self.exclusions = [sum(self.exclusions)]
        self.inclusions = []
        self.masks = []
        return self

    def negate_and_exclude(self):
        self.negate()
        self.exclude()
        return self

class FiberQueryInfo(object):

    r"""
    Information about a processed query

    Attribute
    ---------
        tracts : set
            set of tract indices resulting from the query
        labels : set
            set of labels resulting by the query
        tracts_endpoints : (set, set)
            sets of labels of where the tract endpoints are
    """

    def __init__(self, tracts=None, labels=None, tracts_endpoints=None):
        if tracts is None:
            tracts = set()
        if labels is None:
            labels = set()
        if tracts_endpoints is None:
            tracts_endpoints = (set(), set())
        self.tracts = tracts
        self.labels = labels
        self.tracts_endpoints = tracts_endpoints

    def __getattribute__(self, name):
        if name in (
            'update', 'intersection_update', 'union', 'intersection',
            'difference', 'difference_update'
        ):
            return self.set_operation(name)
        else:
            return object.__getattribute__(self, name)

    def copy(self):
        return FiberQueryInfo(
            self.tracts.copy(), self.labels.copy(),
            (self.tracts_endpoints[0].copy(), self.tracts_endpoints[1].copy()),
        )

    def set_operation(self, name):
        def operation(tract_query_info):
            tracts_op = getattr(self.tracts, name)
            if name == 'intersection':
                name_labels = 'union'
            elif name == 'intersection_update':
                name_labels = 'update'
            else:
                name_labels = name
            labels_op = getattr(self.labels, name_labels)

            new_tracts = tracts_op(tract_query_info.tracts)
            new_labels = labels_op(tract_query_info.labels)

            new_tracts_endpoints = (
                getattr(self.tracts_endpoints[0], name)(
                    tract_query_info.tracts_endpoints[0]
                ),
                getattr(self.tracts_endpoints[1], name)(
                    tract_query_info.tracts_endpoints[1]
                )
            )

            if name.endswith('update'):
                return self
            else:
                return FiberQueryInfo(
                    new_tracts, new_labels,
                    new_tracts_endpoints,
                )

        return operation


# @add_metaclass(DocStringInheritor)
class EvaluateQueries(ast.NodeVisitor):

    r"""
    This class implements the parser to process
    White Matter Query Language modules. By inheriting from
    :py:mod:`ast.NodeVisitor` it uses a syntax close to the
    python language.

    Every node expression visitor has the following signature

    Parameters
    ----------
    node : ast.Node

    Returns
    -------
    tracts : set
        numbers of the tracts that result of this
        query

    labels : set
        numbers of the labels that are traversed by
        the tracts resulting from this query

    """
    __metaclass__ = DocStringInheritor

    relative_terms = [
        'anterior_of',
        'posterior_of',
        'medial_of',
        'lateral_of',
        'inferior_of',
        'superior_of'
    ]

    def __init__(
        self,
        tractography_spatial_indexing,
    ):
        self.tractography_spatial_indexing = tractography_spatial_indexing

        self.evaluated_queries_info = {}
        self.queries_to_save = set()

        self.evaluating_endpoints = False

    def visit_Module(self, node):
        for line in node.body:
            self.visit(line)

    def visit_Compare(self, node):
        if any(not isinstance(op, ast.NotIn) for op in node.ops):
            raise TractQuerierSyntaxError(
                "Invalid syntax in query line %d" % node.lineno
            )

        query_info = self.visit(node.left).copy()
        for value in node.comparators:
            query_info_ = self.visit(value)
            query_info.difference_update(query_info_)

        return query_info

    def visit_BoolOp(self, node):
        query_info = self.visit(node.values[0])
        query_info = query_info.copy()

        if isinstance(node.op, ast.Or):
            for value in node.values[1:]:
                query_info_ = self.visit(value)
                query_info.update(query_info_)

        elif isinstance(node.op, ast.And):
            for value in node.values[1:]:
                query_info_ = self.visit(value)
                query_info.intersection_update(query_info_)

        else:
            return self.generic_visit(node)

        return query_info

    def visit_BinOp(self, node):
        info_left = self.visit(node.left)
        info_right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return info_left.union(info_right)
        if isinstance(node.op, ast.Mult):
            return info_left.intersection(info_right)
        if isinstance(node.op, ast.Sub):
            return (
                info_left.difference(info_right)
            )
        else:
            return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        query_info = self.visit(node.operand)
        if isinstance(node.op, ast.Invert):
            return FiberQueryInfo(
                set(
                    tract for tract in query_info.tracts
                    if (
                        self.tractography_spatial_indexing.
                        crossing_tracts_labels[tract].
                        issubset(query_info.labels)
                    )
                ),
                query_info.labels
            )
        elif isinstance(node.op, ast.UAdd):
            return query_info
        elif isinstance(node.op, ast.USub) or isinstance(node.op, ast.Not):
            all_labels = set(
                self.tractography_spatial_indexing.
                crossing_labels_tracts.keys()
            )
            all_labels.difference_update(query_info.labels)
            all_tracts = set().union(*tuple(
                (
                    self.tractography_spatial_indexing.
                    crossing_labels_tracts[label]
                    for label in all_labels
                )
            ))

            new_info = FiberQueryInfo(all_tracts, all_labels)
            return new_info
        else:
            raise TractQuerierSyntaxError(
                "Syntax error in query line %d" % node.lineno)

    def visit_Str(self, node):
        query_info = FiberQueryInfo()
        for name in fnmatch.filter(self.evaluated_queries_info.keys(), node.s):
            query_info.update(self.evaluated_queries_info[name])
        return query_info

    def visit_Call(self, node):
        # Single string argument function
        if (
            isinstance(node.func, ast.Name) and
            len(node.args) == 1 and
            len(node.keywords) == 0 and
            not hasattr(node, 'starargs') and
            not hasattr(node, 'kwargs')
            ):
            if (node.func.id.lower() == 'only'):
                query_info = self.visit(node.args[0])

                only_tracts = set(
                    tract for tract in query_info.tracts
                    if (
                        self.tractography_spatial_indexing.
                        crossing_tracts_labels[tract].
                        issubset(query_info.labels)
                    )
                )
                only_endpoints = tuple((
                    set(
                        tract for tract in query_info.tracts_endpoints[i]
                        if (
                            self.tractography_spatial_indexing.
                            ending_tracts_labels[i][tract] in query_info.labels
                        )
                    )
                    for i in (0, 1)
                ))
                return FiberQueryInfo(
                    only_tracts,
                    query_info.labels,
                    only_endpoints
                )
            elif (node.func.id.lower() == 'endpoints_in'):
                query_info = self.visit(node.args[0])
                new_tracts = query_info.tracts_endpoints[0].union(query_info.tracts_endpoints[1])
                return FiberQueryInfo(new_tracts, query_info.labels, query_info.tracts_endpoints)
            elif (node.func.id.lower() == 'both_endpoints_in'):
                query_info = self.visit(node.args[0])
                new_tracts = (
                    query_info.tracts_endpoints[0].
                    intersection(query_info.tracts_endpoints[1])
                )
                return FiberQueryInfo(
                    new_tracts, query_info.labels,
                    query_info.tracts_endpoints
                )
            elif (
                node.func.id.lower() == 'save' and
                isinstance(node.args, ast.Str)
            ):
                self.queries_to_save.add(node.args[0].s)
                return
            elif node.func.id.lower() in self.relative_terms:
                return self.process_relative_term(node)

        raise TractQuerierSyntaxError("Invalid query in line %d" % node.lineno)

    def process_relative_term(self, node):
        r"""
        Processes the relative terms

        * anterior_of
        * posterior_of
        * superior_of
        * inferior_of
        * medial_of
        * lateral_of

        Parameters
        ----------
        node :  :py:class:`ast.Node`
            Parsed tree


        Returns
        -------

        tracts, labels

        tracts :  set
            Numbers of the tracts that result of this
            query

        labels :  set
            Numbers of the labels that are traversed by
            the tracts resulting from this query
        """
        if len(self.tractography_spatial_indexing.label_bounding_boxes) == 0:
            return FiberQueryInfo()

        arg = node.args[0]
        if isinstance(arg, ast.Name):
            query_info = self.visit(arg)
        elif isinstance(arg, ast.Attribute):
            if arg.attr.lower() in ('left', 'right'):
                side = arg.attr.lower()
                query_info = self.visit(arg)
        else:
            raise TractQuerierSyntaxError(
                "Attribute not recognized for relative specification."
                "Line %d" % node.lineno
            )

        labels = query_info.labels

        labels_generator = (l for l in labels)

        try:
            bounding_box = (
                self.tractography_spatial_indexing.
                label_bounding_boxes[next(labels_generator)]
            )
            for label in labels_generator:
                bounding_box = bounding_box.union(
                    self.tractography_spatial_indexing.
                    label_bounding_boxes[label]
                )
        except KeyError as e:
            raise TractQuerierLabelNotFound(
                "Label %s not found in atlas file" % e
            )
        function_name = node.func.id.lower()

        name = function_name.replace('_of', '')

        if (
            name in ('anterior', 'inferior') or
            name == 'medial' and side == 'left' or
            name == 'lateral' and side == 'right'
        ):
            operator = gt
        else:
            operator = lt

        if name == 'medial':
            if side == 'left':
                name = 'right'
            else:
                name = 'left'
        elif name == 'lateral':
            if side == 'left':
                name = 'left'
            else:
                name = 'right'

        tract_bounding_box_coordinate =\
            self.tractography_spatial_indexing.tract_bounding_boxes[name]

        tract_endpoints_pos =\
            self.tractography_spatial_indexing.tract_endpoints_pos

        bounding_box_coordinate = getattr(bounding_box, name)

        if name in ('left', 'right'):
            column = 0
        elif name in ('anterior', 'posterior'):
            column = 1
        elif name in ('superior', 'inferior'):
            column = 2

        tracts = set(
            operator(
                tract_bounding_box_coordinate,
                bounding_box_coordinate
            ).nonzero()[0]
        )

        endpoints = tuple((
            set(
                operator(
                    tract_endpoints_pos[:, i, column],
                    bounding_box_coordinate
                ).nonzero()[0]
            )
            for i in (0, 1)
        ))

        labels = set().union(*tuple((
            self.tractography_spatial_indexing.crossing_tracts_labels[tract]
            for tract in tracts
        )))

        return FiberQueryInfo(tracts, labels, endpoints)

    def visit_Assign(self, node):
        if len(node.targets) > 1:
            raise TractQuerierSyntaxError(
                "Invalid assignment in line %d" % node.lineno)

        queries_to_evaluate = self.process_assignment(node)

        for query_name, value_node in queries_to_evaluate.items():
            self.queries_to_save.add(query_name)
            self.evaluated_queries_info[query_name] = self.visit(value_node)

    def visit_AugAssign(self, node):
        if not isinstance(node.op, ast.BitOr):
            raise TractQuerierSyntaxError(
                "Invalid assignment in line %d" % node.lineno)

        queries_to_evaluate = self.process_assignment(node)

        for query_name, value_node in queries_to_evaluate.items():
            query_info = self.visit(value_node)
            self.evaluated_queries_info[query_name] = query_info

    def process_assignment(self, node):
        r"""
        Processes the assignment operations

        Parameters
        ----------
        node :  :py:class:`ast.Node`
            Parsed tree


        Returns
        -------

        queries_to_evaluate: dict
            A dictionary or pairs '<name of the query>'= <node to evaluate>

        """
        queries_to_evaluate = {}
        if 'target' in node._fields:
            target = node.target
        if 'targets' in node._fields:
            target = node.targets[0]

        if isinstance(target, ast.Name):
            queries_to_evaluate[target.id] = node.value
        elif (
            isinstance(target, ast.Attribute) and
            target.attr == 'side'
        ):
            node_left, node_right = self.rewrite_side_query(node)
            self.visit(node_left)
            self.visit(node_right)
        elif (
            isinstance(target, ast.Attribute) and
            isinstance(target.value, ast.Name)
        ):
            queries_to_evaluate[
                target.value.id.lower() + '.'
                + target.attr.lower()] = node.value
        else:
            raise TractQuerierSyntaxError(
                "Invalid assignment in line %d" % node.lineno)
        return queries_to_evaluate

    def rewrite_side_query(self, node):
        r"""
        Processes the side suffixes in a query

        Parameters
        ----------
        node :  :py:class:`ast.Node`
            Parsed tree


        Returns
        -------

        node_left, node_right: nodes
            two AST nodes, one for the query
            instantiated on the left hemisphere
            one for the query instantiated on the right hemisphere

        """
        node_left = deepcopy(node)
        node_right = deepcopy(node)

        for node_ in ast.walk(node_left):
            if isinstance(node_, ast.Attribute):
                if node_.attr == 'side':
                    node_.attr = 'left'
                elif node_.attr == 'opposite':
                    node_.attr = 'right'

        for node_ in ast.walk(node_right):
            if isinstance(node_, ast.Attribute):
                if node_.attr == 'side':
                    node_.attr = 'right'
                elif node_.attr == 'opposite':
                    node_.attr = 'left'

        return node_left, node_right

    def visit_Name(self, node):
        if node.id in self.evaluated_queries_info:
            return self.evaluated_queries_info[node.id]
        else:
            raise TractQuerierSyntaxError(
                "Invalid query name in line %d: %s" % (node.lineno, node.id))

    def visit_Attribute(self, node):
        if not isinstance(node.value, ast.Name):
            raise TractQuerierSyntaxError(
                "Invalid query in line %d: %s" % node.lineno)

        query_name = node.value.id + '.' + node.attr
        if query_name in self.evaluated_queries_info:
            return self.evaluated_queries_info[query_name]
        else:
            raise TractQuerierSyntaxError(
                "Invalid query name in line %d: %s" %
                (node.lineno, query_name)
            )

    def visit_Num(self, node):
        if (
            node.n in
            self.tractography_spatial_indexing.crossing_labels_tracts
        ):
            tracts = (
                self.tractography_spatial_indexing.
                crossing_labels_tracts[node.n]
            )
        else:
            tracts = set()

        endpoints = (set(), set())
        for i in (0, 1):
            elt = self.tractography_spatial_indexing.ending_labels_tracts[i]
            if node.n in elt:
                endpoints[i].update(elt[node.n])

        labelset = set((node.n,))
        tract_info = FiberQueryInfo(
            tracts, labelset,
            endpoints
        )

        return tract_info

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id in self.evaluated_queries_info.keys():
                self.queries_to_save.add(node.value.id)
            else:
                raise TractQuerierSyntaxError(
                    "Query %s not known line: %d" %
                    (node.value.id, node.lineno)
                )
        elif isinstance(node.value, ast.Module):
            self.visit(node.value)
        else:
            raise TractQuerierSyntaxError(
                "Invalid expression at line: %d" % (node.lineno))

    def generic_visit(self, node):
        raise TractQuerierSyntaxError(
            "Invalid Operation %s line: %d" % (type(node), node.lineno))

    def visit_For(self, node):
        id_to_replace = node.target.id.lower()

        iter_ = node.iter
        if isinstance(iter_, ast.Str):
            list_items = fnmatch.filter(
                self.evaluated_queries_info.keys(), iter_.s.lower())
        elif isinstance(iter_, ast.List):
            list_items = []
            for item in iter_.elts:
                if isinstance(item, ast.Name):
                    list_items.append(item.id.lower())
                else:
                    raise TractQuerierSyntaxError(
                        'Error in FOR statement in line %d,'
                        ' elements in the list must be query names' %
                        node.lineno
                    )

        original_body = ast.Module(body=node.body)

        for item in list_items:
            aux_body = deepcopy(original_body)
            for node_ in ast.walk(aux_body):
                if (
                    isinstance(node_, ast.Name) and
                    node_.id.lower() == id_to_replace
                ):
                    node_.id = item

            self.visit(aux_body)


class TractQuerierSyntaxError(ValueError):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TractQuerierLabelNotFound(ValueError):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class RewriteChangeNotInPrescedence(ast.NodeTransformer):

    def visit_BoolOp(self, node):
        predicate = lambda value: not (
            isinstance(value, ast.Compare) and
            isinstance(value.ops[0], ast.NotIn)
        )

        values_which_are_not_in_op = [value for value in takewhile(
            predicate,
            node.values[1:]
        )]

        if (len(values_which_are_not_in_op) == len(node.values) - 1):
            return node

        old_CompareNode = node.values[len(values_which_are_not_in_op) + 1]
        new_CompareNodeLeft = ast.copy_location(
            ast.BoolOp(
                op=node.op,
                values=(
                    [node.values[0]] +
                    values_which_are_not_in_op +
                    [old_CompareNode.left]
                )
            ),
            node
        )

        new_CompareNode = ast.copy_location(
            ast.Compare(
                left=new_CompareNodeLeft,
                ops=old_CompareNode.ops,
                comparators=old_CompareNode.comparators
            ),
            node
        )

        rest_of_the_values = node.values[len(values_which_are_not_in_op) + 2:]

        if len(rest_of_the_values) == 0:
            return self.visit(new_CompareNode)
        else:
            return self.visit(ast.copy_location(
                ast.BoolOp(
                    op=node.op,
                    values=(
                        [new_CompareNode] +
                        rest_of_the_values
                    )
                ),
                node
            ))


class RewritePreprocess(ast.NodeTransformer):

    def __init__(self, *args, **kwargs):
        if 'include_folders' in kwargs:
            self.include_folders = kwargs['include_folders']
            kwargs['include_folders'] = None
            del kwargs['include_folders']
        else:
            self.include_folders = ['.']
        super(RewritePreprocess, self).__init__(*args, **kwargs)

    def visit_Attribute(self, node):
        return ast.copy_location(
            ast.Attribute(
                value=self.visit(node.value),
                attr=node.attr.lower()
            ),
            node
        )

    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(id=node.id.lower()),
            node
        )

    def visit_Str(self, node):
        return ast.copy_location(
            ast.Str(s=node.s.lower()),
            node
        )

    def visit_Import(self, node):
        try:
            module_names = []
            for module_name in node.names:
                file_name = module_name.name
                found = False
                for folder in self.include_folders:
                    file_ = path.join(folder, file_name)
                    if path.exists(file_) and path.isfile(file_):
                        module_names.append(file_)
                        found = True
                        break
                if not found:
                    raise TractQuerierSyntaxError(
                        'Imported file not found: %s' % file_name
                    )
            imported_modules = [
                ast.parse(open(module_name).read(), filename=module_name)
                for module_name in module_names
            ]
        except SyntaxError:
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            formatted_lines = traceback.format_exc().splitlines()
            raise TractQuerierSyntaxError(
                'syntax error in line %s line %d: \n%s\n%s' %
                (
                    module_name,
                    exc_value[1][1],
                    formatted_lines[-3],
                    formatted_lines[-2]
                )
            )

        new_node = ast.Module(imported_modules)

        return ast.copy_location(
            self.visit(new_node),
            node
        )


class EvaluateQueriesVolumetric(EvaluateQueries):
    """
    This class implements the parser to process WMQL modules in volume
    instead of fibers"""

    def __init__(self, labeled_img):
        self.labeled_img = labeled_img
        self.evaluated_queries_info = {}
        self.queries_to_save = set()

    def visit_Compare(self, node):
        if any(not isinstance(op, ast.NotIn) for op in node.ops):
            raise TractQuerierSyntaxError(
                "Invalid syntax in query line %d" % node.lineno
            )

        query_to_include = self.visit(node.left).copy()
        if len(node.comparators) > 1:
            raise NotImplementedError('More than one comparator')

        comparator = node.comparators[0]

        query_to_exclude = self.visit(comparator).copy()
        query_to_exclude.exclude()

        return VolumeQueryInfo(query_to_include.inclusions,
                               query_to_exclude.exclusions + query_to_include.exclusions,
                               query_to_include.seeds)

    def visit_BoolOp(self, node):
        qinfo_node = self.visit(node.values[0]).copy()

        if isinstance(node.op, ast.Or):
            for value in node.values[1:]:
                query_info_ = self.visit(value).copy()
                qinfo_node = qinfo_node.union(query_info_)

        elif isinstance(node.op, ast.And):
            for value in node.values[1:]:
                query_info_ = self.visit(value).copy()
                qinfo_node = qinfo_node.intersection(query_info_)

        return qinfo_node

    def visit_BinOp(self, node):
        info_left = self.visit(node.left).copy()
        info_right = self.visit(node.right).copy()
        if isinstance(node.op, ast.Add):
            return info_left.union(info_right)
        if isinstance(node.op, ast.Mult):
            return info_left.intersect_mask(info_right)
        if isinstance(node.op, ast.Sub):
            return info_left.substract(info_right)
        else:
            return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        query_info = self.visit(node.operand).copy()
        if isinstance(node.op, ast.UAdd):
            return query_info
        elif isinstance(node.op, ast.USub) or isinstance(node.op, ast.Not):
            return query_info.negate()
        else:
            raise TractQuerierSyntaxError(
                "Unary operation {} not found".format(node.op))

    def visit_Call(self, node):
        # Single string argument function
        if (
            isinstance(node.func, ast.Name) and
            len(node.args) == 1 and
            len(node.keywords) == 0 and
            not hasattr(node, 'starargs') and
            not hasattr(node, 'kwargs')
            ):
            if (node.func.id.lower() == 'only'):
                query_info = self.visit(node.args[0]).copy()
                return query_info.negate_and_exclude()
            elif (node.func.id.lower() == 'endpoints_in'):
                query_info = self.visit(node.args[0]).copy()
                return query_info.to_seed_mask()
            elif (node.func.id.lower() == 'both_endpoints_in'):
                query_info = self.visit(node.args[0]).copy()
                return query_info.to_seed_mask()
            elif (
                node.func.id.lower() == 'save' and
                isinstance(node.args, ast.Str)
            ):
                self.queries_to_save.add(node.args[0].s)
                return
            elif node.func.id.lower() in self.relative_terms:
                return self.process_relative_term(node)

        raise TractQuerierSyntaxError("Invalid query in line %d" % node.lineno)

    def process_relative_term(self, node):
        r"""
        Processes the relative terms

        * anterior_of
        * posterior_of
        * superior_of
        * inferior_of
        * medial_of
        * lateral_of

        Parameters
        ----------
        node :  :py:class:`ast.Node`
            Parsed tree


        Returns
        -------

        tracts, labels

        tracts :  set
            Numbers of the tracts that result of this
            query

        labels :  set
            Numbers of the labels that are traversed by
            the tracts resulting from this query
        """
        arg = node.args[0]

        if isinstance(arg, ast.Name):
            query_info = self.visit(arg).copy()
        elif isinstance(arg, ast.Attribute):
            if arg.attr.lower() in ('left', 'right'):
                side = arg.attr.lower()
                query_info = self.visit(arg).copy()
        else:
            raise TractQuerierSyntaxError(
                "Attribute not recognized for relative specification."
                "Line %d" % node.lineno
            )

        if len(query_info.inclusions) > 1 or query_info.exclusions or\
           len(query_info.seeds) > 1:
            raise ValueError("Relative terms should be applied to masks")

        mask = query_info.inclusions[0]
        nzr = mask.nonzero()

        function_name = node.func.id.lower()
        name = function_name.replace('_of', '')

        # ASSUMING RPI
        if (
            name in ('anterior', 'superior') or
            name == 'medial' and side == 'right' or
            name == 'lateral' and side == 'left'
        ):
            boundary = max
        else:
            boundary = min

        if name in ['medial', 'lateral']:
            name = side

        if name in ('left', 'right'):
            column = 0
        elif name in ('anterior', 'posterior'):
            column = 1
        elif name in ('inferior', 'superior'):
            column = 2

        lim = boundary(nzr[column])

        if boundary == max:
            lower = lim+1
            upper = None
        else:
            lower = None
            upper = lim

        slicing = [slice(None) for _ in range(3)]
        slicing[column] = slice(lower, upper, None)

        new_mask = mask*0
        new_mask[tuple(slicing)] = 1

        return VolumeQueryInfo([new_mask])

    def visit_Num(self, node):
        import numpy as np
        mask = (self.labeled_img == node.n).astype(np.bool)
        if not mask.any():
            return VolumeQueryInfo([0])
        return VolumeQueryInfo([mask])

    def visit_Str(self, node):
        query_info = VolumeQueryInfo()

        for name in fnmatch.filter(self.evaluated_queries_info.keys(), node.s):
            query_info = query_info.intersection(self.evaluated_queries_info[name])
        return query_info

def queries_preprocess(query_file, filename='<unknown>', include_folders=[]):

    try:
        query_file_module = ast.parse(query_file) # , filename='<unknown>')
    except SyntaxError:
        import sys
        import traceback
        filename= query_file # This was missing in interpreting error
        exc_type, exc_value, exc_traceback = sys.exc_info()
        formatted_lines = traceback.format_exc().splitlines()
        raise TractQuerierSyntaxError(
            'syntax error in line %s line %d: \n%s' %
            (
                filename,
                exc_value.lineno,
                # The offset should not be necessary
                # If you really need that, use
                # exc_value.offset
                exc_value.text
            )
        )

    rewrite_preprocess = RewritePreprocess(include_folders=include_folders)
    rewrite_precedence_not_in = RewriteChangeNotInPrescedence()

    preprocessed_module = rewrite_precedence_not_in.visit(
        rewrite_preprocess.visit(query_file_module)
    )

    return preprocessed_module.body


def eval_queries(
    query_file_body,
    tractography_spatial_indexing
):
    eq = EvaluateQueries(tractography_spatial_indexing)

    if isinstance(query_file_body, list):
        eq.visit(ast.Module(query_file_body))
    else:
        eq.visit(query_file_body)

    return dict([
        (key, eq.evaluated_queries_info[key].tracts)
        for key in eq.queries_to_save
    ])


def eval_queries_volume(query_file_body, labels_img):
    eq = EvaluateQueriesVolumetric(labels_img)

    if isinstance(query_file_body, list):
        eq.visit(ast.Module(query_file_body))
    else:
        eq.visit(query_file_body)

    return dict([
        (key, eq.evaluated_queries_info[key])
        for key in eq.queries_to_save
    ])


def queries_syntax_check(query_file_body):
    class DummySpatialIndexing:

        def __init__(self):
            self.crossing_tracts_labels = {}
            self.crossing_labels_tracts = {}
            self.ending_tracts_labels = ({}, {})
            self.ending_labels_tracts = ({}, {})
            self.label_bounding_boxes = {}
            self.tract_bounding_boxes = {}

    eval_queries(query_file_body, DummySpatialIndexing())


def labels_for_tracts(crossing_tracts_labels):
    crossing_labels_tracts = {}
    for i, f in crossing_tracts_labels.items():
        for l in f:
            if l in crossing_labels_tracts:
                crossing_labels_tracts[l].add(i)
            else:
                crossing_labels_tracts[l] = set((i,))
    return crossing_labels_tracts
