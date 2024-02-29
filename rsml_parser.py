from pathlib import Path
import numpy as np
import pandas as pd
import csv
import itertools
import os
import xml.etree.ElementTree as ET
import argparse


###### parsing
def parse_rsml(rsml_path):
    tree = ET.parse(rsml_path)
    root = tree.getroot()
    # then root[0][0].tag .attrib or .text
    return root

#### dictionnary creation from xml data
def extract_plantroot_data_from_plantroot_node(plantroot_node):
    assert plantroot_node.tag == 'root'
    # get plant root name and keys
    plantroot_name = plantroot_node.attrib['ID']
    keys_list = plantroot_node[0][0][0].keys()
    plantroot_data = {k:[] for k in keys_list  }
    for point_node in plantroot_node[0][0]:
        for k in keys_list:
            plantroot_data[k].append(float(point_node.attrib[k]))
    return {plantroot_name : plantroot_data }


def extract_plantroots_data_from_rootnode(root):
    plantroot_nodes = itertools.chain(
        root[1][0][0:], # main plantroot 1
        root[1][1][0:], # main plantroot 2
        root[1][0][0][1:], # lateral plantroots of 1
        root[1][1][0][1:] # lateral plantroots of 2
        )
    results = {}
    for node in plantroot_nodes:
        res = extract_plantroot_data_from_plantroot_node(node)
        results.update(res)
    return results



#### dataframe creation

def replace_single_element_list(cell):
    if isinstance(cell, list) and len(cell) == 1:
        return cell[0]
    else:
        return cell



def add_prefix_except_column(df, prefix, column_to_exclude):
    names_map = { n : prefix  + n  for n in  df.columns if n != column_to_exclude }
    dfout = df.rename(columns=names_map)
    return dfout

def unique_merge_per_column(df, column ='coord_t'):
    df = df.groupby(column).agg(lambda x: x.dropna().tolist()).reset_index()
    df = df.applymap(replace_single_element_list) # remove bracket one liste contains one single element
    return df

def merge_two_plantroot_dfs(df1,df2, column = 'coord_t'):
    # Fusion des dataframes sur la colonne 'time' en utilisant une union
    merged_df = pd.merge(df1, df2, on=column, how='outer')
    return merged_df

def merge_list_of_plantroot_dfs(dfs, column ='coord_t'):
    dfout = dfs[0]
    for df in dfs[1:]:
        dfout = pd.merge(dfout, df, on=column, how='outer')
        dfout = dfout.sort_values(by=column)
    return dfout
    
### compute length
def compute_plantroot_len(df): 
    x =  df[['coord_x', 'coord_y']].to_numpy() # shape (len, 2)
    diffs = x[1:] - x[:-1] # compute differences
    diffs = np.concatenate((np.array([[0.,0.]]), diffs)) # add 0 a time 0
    distances = np.sum(diffs**2,axis=1)**0.5 # compute norm from differences
    cumulative_distances = distances.cumsum()
    return cumulative_distances

def add_plantroot_len_column(df, column_name='cumuldist'):
    cum = compute_plantroot_len(df)
    df.insert(1, column_name, cum , True )
    return df


#### MAIN 
def main():
    #parsing
    parser = argparse.ArgumentParser(description='rsml parser and converter')
    parser.add_argument('inputfile', 
                        type=str, 
                        help='path of rsml file'
    )

    parser.add_argument('--column-to-merge', 
                        default='coord_t',
                        type=str, 
                        help='name of the column to merge between plantroot data frame'
    )
    parser.add_argument('--columns-to-drop', 
                        default=['diameter', 'vx', 'vy', 'coord_th', 'coord_x', 'coord_y'], # or ['diameter', 'vx', 'vy']
                        nargs='+',
                        type=str, 
                        help='name of the column to drop within each plantroot data frame'
    )

    parser.add_argument('--save-subfiles', 
                        action='store_true',
                        help='option to store intermediate plantroot files'
    )

    args = parser.parse_args()


    # manage path
    input_file_path = Path(args.inputfile)
    output_file_path = input_file_path.with_suffix('.csv')
    subfiles_folder_path = Path(input_file_path.parent,input_file_path.stem)

    # manage options
    column_merge = args.column_to_merge
    columns_to_drop = args.columns_to_drop
    save_subfiles = args.save_subfiles

    # parse rsml
    root = parse_rsml(input_file_path)

    # intialise result dictionnary

    data = extract_plantroots_data_from_rootnode(root)
    #times = extract_ordered_times_frames(data)

    # create dataframes from data dict
    dfs = [pd.DataFrame(data[k]) for k in data]  


    # compute lenght for each data 
    dfs = [ add_plantroot_len_column(df) for df in dfs ]

    # remove unecessary data from dataframes
    dfs = [ df.drop(columns = columns_to_drop )  for df in dfs ] 


    # gather data per time
    dfs = [ unique_merge_per_column(df) for df in dfs]

    # add prefix to columns
    dfs = [add_prefix_except_column(df,k + ' ', column_to_exclude= column_merge ) for (df,k) in zip(dfs,data) ]

    # merge dfs     
    df = merge_list_of_plantroot_dfs(dfs, column = column_merge)

    # save to file
    df.to_csv(output_file_path,index=False)

    if save_subfiles: 
        subfiles_folder_path.mkdir(exist_ok=True, parents=True)
        subnames = [k.replace('.','_') for k in data.keys()] # replace dot per underscores
        for (k, sdf) in zip(subnames, dfs):
            #import pdb; pdb.set_trace()
            subfile_path =  Path(subfiles_folder_path, k + '.csv')
            sdf.to_csv(subfile_path, index=False)



    import pdb; pdb.set_trace()


if __name__=='__main__':
    main()


