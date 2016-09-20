'''
Created on Apr 7, 2016

@author: Andrew Roth
'''
import dollo.trees
import gzip
import pandas as pd
import pickle


def load_tree(file_name):
    
    with gzip.GzipFile(file_name, 'r') as fh:
        tree = pickle.load(fh)
    
    return tree


def tree_to_table(tree):
    tree_table = []

    for n in tree.nodes:
        for child in n.children:
            child.parent = n.label
    
    for n in tree.nodes:
        node_attr = {}

        for key, value in n.__dict__.iteritems():
            if isinstance(value, (int, float, str)):
                node_attr[key] = value

        tree_table.append(node_attr)
        
    return pd.DataFrame(tree_table)


def table_to_tree(table):
    
    table = table.sort_values(by='parent')

    nodes = {}
    
    skip = table.isnull()
    
    for idx in table.index:
        n = dollo.trees.TreeNode()
            
        for col in table.columns:
            if not skip.loc[idx, col]:
                setattr(n, col, table.loc[idx, col])
                
        nodes[n.label] = n
    
    for label, node in nodes.iteritems():
        if hasattr(node, 'parent') and node.parent >= 0:
            nodes[node.parent].children.append(node)

    return nodes[0]