# native packages
from pathlib import Path
import itertools
import xml.etree.ElementTree as ET
import argparse
# packages to install
import numpy as np
import pandas as pd


###### parsing
def parse_rsml(rsml_path):
    tree = ET.parse(rsml_path)
    root = tree.getroot()
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

def add_prefix_except_column(df, prefix, column_to_exclude):
    names_map = { n : prefix  + n  for n in  df.columns if n != column_to_exclude }
    dfout = df.rename(columns=names_map)
    return dfout

def replace_single_element_list(cell, keep_last_point_data_only= False):
    if isinstance(cell, list) and len(cell) == 1:
        return cell[0]
    elif isinstance(cell, list) and keep_last_point_data_only:
        return cell[-1] 
    else:
        return cell

def unique_merge_per_column(df, column ='coord_t', keep_last_point_data_only=False):
    df = df.groupby(column).agg(lambda x: x.dropna().tolist()).reset_index()
    df = df.applymap(lambda x: replace_single_element_list(x, keep_last_point_data_only=keep_last_point_data_only )) # remove bracket one liste contains one single element
    return df

def merge_list_of_plantroot_dfs(dfs, column ='coord_t'):
    dfout = dfs[0]
    for df in dfs[1:]: 
        dfout = pd.merge(dfout, df, on=column, how='outer')
        dfout = dfout.sort_values(by=column)
    return dfout


def remove_fractional_frames_in_df(df,column='coord_t'):
    return df[df[column].astype(int) == df[column]]

    
### compute length
def compute_plantroot_len(df , scale_factor=1.0): 
    x =  df[['coord_x', 'coord_y']].to_numpy() # shape (len, 2)
    diffs = x[1:] - x[:-1] # compute differences
    diffs = np.concatenate((np.array([[0.,0.]]), diffs)) # add 0 a time 0
    distances = np.sum(diffs**2,axis=1)**0.5 # compute norm from differences
    distances = distances * scale_factor
    cumulative_distances = distances.cumsum()
    return cumulative_distances

def add_plantroot_len_column(df, column_name='cumuldist', scale_factor=1.0):
    cumulative= compute_plantroot_len(df, scale_factor=scale_factor)
    df.insert(7, column_name, cumulative , True )
    return df

#### MAIN 
def main():
    #parsing
    parser = argparse.ArgumentParser(description='rsml parser and converter')
    parser.add_argument('inputfile', 
                        type=str, 
                        help='path of rsml file'
    )


    parser.add_argument('--columns-to-drop', 
                        default=['diameter', 'vx', 'vy', 'coord_x', 'coord_y'], # or ['diameter', 'vx', 'vy']
                        nargs='+',
                        type=str, 
                        help='name of the column to drop within each plantroot data frame'
    )

    parser.add_argument('--scale-factor', 
                        default = 1.0,
                        type=float,
                        help='distance factor to multiply distance'
    )

    parser.add_argument('--save-subfiles', 
                        action='store_true',
                        help='option to store intermediate plantroot files'
    )
    
    parser.add_argument('--remove-fractional-frames', 
                        action='store_true',
                        help='remove lines where fractionnal point (non integer coord_t value)'
    )

    parser.add_argument('--keep-last-point-data-only', 
                        action='store_true',
                        help='for a given time frame, keep only last point data'
    )



    args = parser.parse_args()


    # manage path
    input_file_path = Path(args.inputfile)
    output_file_path = input_file_path.with_suffix('.csv')
    subfiles_folder_path = Path(input_file_path.parent,input_file_path.stem)

    # manage options
    column_merge = 'coord_t' # hardcoded
    columns_kept_only_once = ['coord_th']
    columns_to_drop = args.columns_to_drop
    save_subfiles = args.save_subfiles
    scale_factor = args.scale_factor
    remove_fractional_frames = args.remove_fractional_frames
    keep_last_point_data_only = args.keep_last_point_data_only

    # parse rsml
    root = parse_rsml(input_file_path)

    # intialise result dictionnary
    data = extract_plantroots_data_from_rootnode(root)

    # create dataframes from data dict
    dfs = [pd.DataFrame(data[k]) for k in data]  

    # compute lenght for each data 
    dfs = [add_plantroot_len_column(df, scale_factor=scale_factor) for df in dfs]

    # remove unecessary data from dataframes
    
    dfs = [dfs[0]] + [df.drop(columns = columns_kept_only_once ) for df in dfs[1:]] if len(dfs) > 1 else dfs
    dfs = [df.drop(columns = columns_to_drop ) for df in dfs] 
    
    
    # gather data per time
    dfs = [unique_merge_per_column(df, keep_last_point_data_only= keep_last_point_data_only) for df in dfs]

    # remove fractional frames if necessary
    if remove_fractional_frames:
        dfs = [remove_fractional_frames_in_df(df) for df in dfs]

    # add prefix to columns
    dfs = [add_prefix_except_column(df, k + ' ', column_to_exclude= column_merge ) for (df,k) in zip(dfs,data) ]

    # gather subroots of plantroot 1 and 2
    dfs1 =  [ dfs[i] for (i,k) in enumerate(data.keys()) if k.startswith('1')  ]
    dfs2 =  [ dfs[i] for (i,k) in enumerate(data.keys()) if k.startswith('2')  ]

    # merge dfs     
    df1 = merge_list_of_plantroot_dfs(dfs1, column = column_merge)
    df2 = merge_list_of_plantroot_dfs(dfs2, column = column_merge)
    df = merge_list_of_plantroot_dfs(dfs, column = column_merge)

    # save to files
    df.to_csv(output_file_path,index=False)

    import pdb; pdb.set_trace()
    if save_subfiles: 
        subfiles_folder_path.mkdir(exist_ok=True, parents=True)

        # save plantroot 1 and 2
        df1.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root1.csv'), index=False)
        df2.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root2.csv'), index=False)


        # all root subfiles
        subnames = [k.replace('.','_') for k in data.keys()] # replace dot per underscores
        for (k, sdf) in zip(subnames, dfs):
            
            subfile_path =  Path(subfiles_folder_path, k + '.csv')
            sdf.to_csv(subfile_path, index=False)

    print('end of parsing and conversion')


if __name__=='__main__':
    main()


