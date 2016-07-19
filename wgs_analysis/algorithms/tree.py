import numpy as np


def align_trees(tree_1, tree_2):
    """ Align two trees if possible.

    Args:
        tree_1 (dollo.trees.TreeNode): first tree to align
        tree_2 (dollo.trees.TreeNode): second tree to align

    Returns:
        list of tuple: matching between node labels in `tree_1` to node labels in `tree_2`

    Raises:
        ValueError: trees are not alignable

    Align trees if possible and return a mapping between node labels.
    Leaves are matched by leaf name.  Internal nodes are matched if they
    contain their set of descendent leaf names is equal.

    """

    if len(list(tree_1.nodes)) != len(list(tree_2.nodes)):
        raise ValueError('Trees not alignable')

    node_alignment = list()

    for node_1 in tree_1.nodes:

        leaf_set_1 = set([leaf.name for leaf in node_1.leaves])

        matched_node_2 = None
        for node_2 in tree_2.nodes:
            leaf_set_2 = set([leaf.name for leaf in node_2.leaves])
            if leaf_set_1 == leaf_set_2:
                matched_node_2 = node_2

        if matched_node_2 is None:
            raise ValueError('Trees not alignable')

        node_alignment.append((node_1.label, matched_node_2.label))

    return node_alignment


def simplify_tree(tree, node_labels):
    """ Simplify tree based on subset of nodes.

    Args:
        tree (dollo.TreeNode): tree to simplify
        node_labels (list): node labels for subset of nodes

    Returns:
        dollo.TreeNode: simplified trees with redundant nodes/branches removed

    """

    # Filter redundant branches
    while True:
        altered = False
        for node in tree.nodes:
            new_children = list()
            for child in node.children:
                if child.label not in node_labels:
                    new_children.extend(child.children)
                    altered = True
                else:
                    new_children.append(child)
            node.children = new_children
            if altered:
                break
        if not altered:
            break

    while True:
        if tree.label not in node_labels and len(tree.children) == 1:
            tree = tree.children[0]
        else:
            break

    return tree


def simplify_tree_by_genotype(tree, genotypes):
    """ Simplify tree based on given genotypes.

    Args:
        tree (dollo.TreeNode): tree to simplify
        genotypes (pandas.DataFrame): node table for genotypes

    Returns:
        dollo.TreeNode: simplified trees with redundant nodes/branches removed

    """

    # Add genotypes to nodes
    for node in tree.nodes:
        node.genotype = set(
            genotypes.loc[
                (genotypes['node'] == node.label) &
                (genotypes['ml_presence'] == 1),
                'cluster_id'
            ].values
        )

    # Filter redundant branches
    while True:
        altered = False
        for node in tree.nodes:
            new_children = list()
            for child in node.children:
                if node.genotype == child.genotype:
                    new_children.extend(child.children)
                    altered = True
                else:
                    new_children.append(child)
            node.children = new_children
            if altered:
                break
        if not altered:
            break

    while True:
        if len(tree.genotype) == 0:
            if len(tree.children) > 1:
                raise Exception('no ancestral genotype')
            tree = tree.children[0]
        else:
            break

    return tree


def classify_tree(tree):
    """ Classify tree type.

    Args:
        tree (dollo.TreeNode): tree to classify

    Returns:
        str: classification as 'pure', 'branched', 'chain'

    """

    if len(list(tree.nodes)) == 1:
        return 'pure'

    for node in tree.nodes:
        if len(node.children) > 1:
            return 'branched'

    return 'chain'


def uniform_transition(state, child_state):
    """ Uniform transition score

    Args:
        state (int): state of parent
        child_state (int): state of child

    Returns:
        int: score

    """

    if state == child_state:
        return 0
    else:
        return 1


def calculate_score_recursive(node, f_transition_score):
    """ Tree recursion for maximum parsimony.

    Args:
        node (dollo.TreeNode): node for recursive score calculation
        f_transition_score (callable): state transition function

    Node variables:
        state_scores (numpy.array): array of scores for each state
        state_backtrack (list of list): array of minimizing states for each parent state

    Assumes state_scores are valid at leaves.  At return, state_scores valid at
    all nodes, and state_backtrack valid at all non-root nodes.

    """

    if not node.is_leaf:

        node.state_scores = np.repeat(0., len(node.state_scores))

        for child in node.children:
            calculate_score_recursive(child, f_transition_score)
            child.state_backtrack = [[]] * len(node.state_scores)

        for state in xrange(len(node.state_scores)):

            for child in node.children:

                child_state_scores = np.repeat(0., len(node.state_scores))

                for child_state in xrange(0, len(node.state_scores)):

                    transition_score = f_transition_score(state, child_state)
                    child_state_scores[child_state] = transition_score + child.state_scores[child_state]

                min_score = np.amin(child_state_scores)

                child.state_backtrack[state] = np.where(min_score == child_state_scores)[0]
                node.state_scores[state] += min_score


def backtrack_state_recursive(node, state=None):
    """ Tree backtracking for maximum parsimony state.

    Args:
        node (dollo.TreeNode): node for recursive score calculation

    KwArgs:
        state (int): state assignment for node

    Node variables:
        state_scores (numpy.array): array of scores for each state
        state_backtrack (list of list): array of minimizing states for each parent state

    Assumes state_scores and state_backtrack are valid for all nodes.  At return,
    each node has a state variable representing state for one maximum parsimony
    solution.    

    """

    if state is None:
        node.state = np.argmin(node.state_scores)
    else:
        node.state = state

    for child in node.children:
        backtrack_state_recursive(child, child.state_backtrack[node.state][0])


