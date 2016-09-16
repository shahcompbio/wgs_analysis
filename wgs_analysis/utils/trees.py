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
        tree_table.append(
            {
                'label' : n.label, 
                'name' : n.name, 
                'parent' : getattr(n, 'parent', -1)
            }
        )
        
    return pd.DataFrame(tree_table)

def table_to_tree(table):
    
    table = table.sort_values(by='parent')

    nodes = {}
    
    for _, row in table.iterrows():
        
        n = dollo.trees.TreeNode()
            
        if row['parent'] in nodes:
            n.add_parent(row['parent'])
            
            nodes[row['parent']].children.append(n)
            
        n.name = row['name']
    
        n.label = row['label']
        
        nodes[n.label] = n

    return nodes[0]