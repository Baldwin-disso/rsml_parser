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

##### Compute section numpy thingy
# compute length
def compute_plantroot_len(x , scale_factor=1.0):    
    diffs = x[1:] - x[:-1] # compute differences
    diffs = np.concatenate((np.array([[0.,0.]]), diffs)) # add 0 a time 0
    distances = np.sum(diffs**2,axis=1)**0.5 # compute norm from differences
    distances = distances * scale_factor
    cumulative_distances = distances.cumsum()
    return cumulative_distances

# compute tortuosity

def compute_plantroot_tortuosity(x):
    # x is N times 2 ( coord_x, coord_y) 
    cumulative_distances = compute_plantroot_len(x , scale_factor=1.0)
    straight_diffs = x - x[0]
    straight_distances = np.sum(straight_diffs**2,axis=1)**0.5
    mask = straight_distances == 0
    tortuosities = cumulative_distances/ np.where(mask, 1, straight_distances)
    return tortuosities


# compute cumulative dist until last root junction
def compute_mainplantroot_len_until_last_junction_at_time(t, main_df, lat_dfs, scale_factor=1.0):
    # retrieve times and coordinates of the first point of each lateral root 

    main_points  = main_df.iloc[:, main_df.columns.isin(['coord_t', 'coord_x', 'coord_y']) ].to_numpy()
    lat_first_points = np.array([
        df.iloc[0, df.columns.isin(['coord_t', 'coord_x', 'coord_y']) ].to_numpy() 
        for df in lat_dfs  
    ])

    
    ## 1 filter lateral roots that appears before time t
    ## and keep points from main root that also appear before time t
    lat_first_points = np.array([x for x in lat_first_points if x[0] <= t  ])
    main_points = np.array([x for x in main_points if x[0] <= t  ])
    # case where no lateral root exist at that time
    if lat_first_points.size == 0: 
        return 0,0
    # compute number of laterals roots at time t
    nb_lats_at_t = lat_first_points.shape[0]
    
    
    ## 2 get the last lateral root to consider :
    # sort lateral root by time
    indices = np.argsort(lat_first_points[:, 0], kind='mergesort')[::-1] # STABLE sort with descending order
    lat_first_points = lat_first_points[indices]
    #filter to  the most recent lateral roots only
    lat_first_points = np.array([ x for x in lat_first_points if x[0]==lat_first_points[0,0] ])

    ## 3 if several are found, retrieve the deepest (higher y value)
    indices = np.argsort(lat_first_points[:, 2], kind='mergesort')[::-1] # STABLE sort with descending order
    lat_first_points = lat_first_points[::-1]

    focus_lat_point = lat_first_points[0]
   

    ## 4 find the 2 closest point in main root, take the first one 
    # compute norm between focus point from lateral root to every point of primary root
    d = np.linalg.norm(main_points[:,1:] - focus_lat_point[1:], axis=1) 
    # compute points with the two smallest distances
    smallest_d_indices = np.argsort(d)[:2]
    # sort the indices in order to take the first one that appeared in main
    smallest_d_indices  = np.sort(smallest_d_indices)
    selected_index = smallest_d_indices[0]

    ## truncate main root up to the selected index (included), and compute len
    main_points = main_points[:selected_index+1]
    truncated_len = (
        compute_plantroot_len(main_points[:,1:],scale_factor=scale_factor)[-1] 
        + d[selected_index]*scale_factor
    )
    

    return truncated_len, nb_lats_at_t
    

def compute_mainplantroot_len_until_last_junction(main_df, lat_dfs, scale_factor=1.0):
    # retrieve time frames and loop on them
    main_times = main_df.loc[:, 'coord_t'].to_numpy()
    
    len_until_junction = []
    nb_lateral_plantroot = []
    for t in main_times: # loop over the times of main root data
        l, n = compute_mainplantroot_len_until_last_junction_at_time(
            t,
            main_df,
            lat_dfs,
            scale_factor=scale_factor
        )
        len_until_junction.append(l)
        nb_lateral_plantroot.append(n)

    return len_until_junction, nb_lateral_plantroot




#### MAIN 
def _main(input_file_path, **kwargs):   
    print('parsing and converting : {}'.format(input_file_path))
    # manage path
    input_file_path = Path(input_file_path)
    output_file_path = input_file_path.with_suffix('.csv')
    subfiles_folder_path = Path(input_file_path.parent,input_file_path.stem)

    # manage options
    column_merge = 'coord_t' # hardcoded
    columns_kept_only_once = ['coord_th']
    columns_to_drop = kwargs['columns_to_drop']
    save_subfiles = kwargs['save_subfiles']
    scale_factor = kwargs['scale_factor']
    remove_fractional_frames = kwargs['remove_fractional_frames']
    keep_last_point_data_only = kwargs['keep_last_point_data_only']
    
    # fixed column names 
    COLUMNS = ['cumuldist', 'tortuosity', 'cumuldistjunction', 'nblaterals']

    # parse rsml
    root = parse_rsml(input_file_path)

    # intialise result dictionnary
    data = extract_plantroots_data_from_rootnode(root)

    # create dataframes from data dict
    dfs = { k:pd.DataFrame(data[k]) for k in data}  


    ## ALL kind of computation HERE
    # compute lenght for each data 

    # computation made independantly on each df
    for k in dfs :
        df = dfs[k]
        x = df[['coord_x', 'coord_y']].to_numpy()
        cumuldist = compute_plantroot_len(x, scale_factor=scale_factor)
        tortuosity = compute_plantroot_tortuosity(x)
        df.insert(7, COLUMNS[0], cumuldist , True )
        df.insert(8, COLUMNS[1], tortuosity , True )


    
    # compute len until last junction
    main_df1 =  dfs['1.1'] 
    main_df2 = dfs['2.1']
    lat_dfs1 = [ dfs[k] for k in dfs if k.startswith('1') and k!='1.1']
    lat_dfs2 = [ dfs[k] for k in dfs if k.startswith('2') and k!='2.1' ]
    luj1, nb_lat1 = compute_mainplantroot_len_until_last_junction(
        main_df1, 
        lat_dfs1, 
        scale_factor=scale_factor
    )
    luj2, nb_lat2 = compute_mainplantroot_len_until_last_junction(
        main_df2,
        lat_dfs2, 
        scale_factor=scale_factor
        )
    
    main_df1.insert(7, COLUMNS[2], luj1 , True )
    main_df1.insert(8, COLUMNS[3], nb_lat1 , True )
    main_df2.insert(7, COLUMNS[2], luj2 , True )
    main_df2.insert(8, COLUMNS[3], nb_lat2 , True )

    dfs.update({'1.1':main_df1, '2.1':main_df2 })


    
    ## Now end of column computation : 
    # remove unecessary data from dataframes
    dfs.update( {k:dfs[k].drop(columns = columns_kept_only_once ) for k in list(dfs.keys())[1:]} )
    dfs = { k:dfs[k].drop(columns = columns_to_drop ) for k in dfs }
    
    
    # gather data per time
    dfs = {k:unique_merge_per_column(dfs[k], keep_last_point_data_only= keep_last_point_data_only) for k in dfs}

    # remove fractional frames if necessary
    if remove_fractional_frames:
        dfs = {k:remove_fractional_frames_in_df(dfs[k]) for k in dfs}

    # add prefix to columns
    dfs = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in dfs }

    # gather subroots of plantroot 1 and 2
    dfs1 =  {k:dfs[k] for k in dfs if k.startswith('1')  }
    dfs2 =  {k:dfs[k] for k in dfs if k.startswith('2')  }

    # merge dfs     
    df1 = merge_list_of_plantroot_dfs(list(dfs1.values()), column = column_merge)
    df2 = merge_list_of_plantroot_dfs(list(dfs2.values()), column = column_merge)
    df = merge_list_of_plantroot_dfs(list(dfs.values()), column = column_merge)

    # save to files
    df.to_csv(output_file_path,index=False)

    
    if save_subfiles: 
        subfiles_folder_path.mkdir(exist_ok=True, parents=True)

        # save plantroot 1 and 2
        df1.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root1.csv'), index=False)
        df2.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root2.csv'), index=False)


        for k in dfs:
            subfile_path =  Path(subfiles_folder_path, k.replace('.','_') + '.csv')
            dfs[k].to_csv(subfile_path, index=False)


def main(*args, **kwargs):
    for arg in args: 
        _main(arg, **kwargs)



if __name__=='__main__':
    #parsing
    parser = argparse.ArgumentParser(description='rsml parser and converter')
    parser.add_argument('inputfiles', 
                        type=str, 
                        nargs='+', 
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



    print('rsml_parser with inputs : \n {} '.format(args))

    
    main(
        *args.inputfiles,
        columns_to_drop = args.columns_to_drop,
        scale_factor = args.scale_factor,
        save_subfiles = args.save_subfiles,
        remove_fractional_frames = args.remove_fractional_frames,
        keep_last_point_data_only = args.keep_last_point_data_only
    )

    print('end of parsing and conversion')




