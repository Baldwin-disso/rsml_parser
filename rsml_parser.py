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
    dfout = dfs[0].copy()
    for df in dfs[1:]: 
        dfout = pd.merge(dfout, df, on=column, how='outer')
        #qprint(f"dfout shape : {dfout.shape} ; df shape : {df.shape} " )
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
def compute_mainplantroot_len_until_last_junction_at_frame(f, mdf, lat_dfs, scale_factor=1.0): 

    # retrieve times and coordinates up to current frame
    main_points  = mdf.iloc[:f+1, mdf.columns.isin(['coord_t', 'coord_x', 'coord_y']) ].to_numpy()
    t = main_points[f,0].item() # retrieve current time (the one of last frame)
    

    depth = main_points[f,2].item() # retrieve current depth
    lat_first_points = np.array([
        df.iloc[0, df.columns.isin(['coord_t', 'coord_x', 'coord_y']) ].to_numpy() 
        for df in lat_dfs.values()  
    ])

    
    ## 1 filter lateral roots that appears before time t
    ## and keep points from main root that also appear before time t 
    lat_first_points = np.array([x for x in lat_first_points if x[0] <= t])
    main_points = np.array([x for x in main_points if x[0] <= t  ])
    # case where no lateral root exist at that time
    if lat_first_points.size == 0: 
        return 0,0
    
    # compute number of laterals roots at frame f
    nb_lats_at_f = lat_first_points.shape[0]
    
    
    ## 2 get the deepest lateral root to consider :
    # sort lateral root by depths
    
    indices = np.argsort(lat_first_points[:, 2], kind='mergesort')[::-1] # STABLE sort with descending order along with coord_y (2)
    focus_lat_point = lat_first_points[indices[0]] # consider deepest, this is the one where one should focus
   

    ## 3 find the 2 closest point in main root, take the first one 
    relative_depths_to_focus =  main_points[:,2] -  focus_lat_point[2] # positive are main root plant node deeper than junction 
    neg_rel_depths = relative_depths_to_focus[relative_depths_to_focus < 0] # filter only negatif relative depths
    max_neg_rel_depths = neg_rel_depths.max() # keep max of negative (first before junction)
    selected_index = np.where(relative_depths_to_focus == max_neg_rel_depths)[0][0] # get index of it

    ## truncate main root up to the selected index (included), and compute len
    main_points = main_points[:selected_index+1]

    truncated_len = (
        compute_plantroot_len(main_points[:,1:],scale_factor=scale_factor)[-1] 
        +  np.linalg.norm(focus_lat_point[1:] - main_points[-1,1:] )*scale_factor
    )
    
    return truncated_len, nb_lats_at_f
    

def compute_mainplantroot_len_until_last_junction(mdf, lat_dfs, scale_factor=1.0):
    # retrieve time frames and loop on them
    main_nb_frames = len(mdf)

    len_until_junction = []
    nb_lateral_plantroot = []
    for f in range(main_nb_frames): # loop over the times of main root data
        l, n = compute_mainplantroot_len_until_last_junction_at_frame(
            f,
            mdf,
            lat_dfs,
            scale_factor=scale_factor
        )
        len_until_junction.append(l)
        nb_lateral_plantroot.append(n)

    return len_until_junction, nb_lateral_plantroot



def gather_dataframes_for_sections(mdf, lat_dfs):
    """
        gather data across 3 section,
        dividing main df into 3 main dfs and 
        gathering lateral roots to each of the 3 according to :
        - section 1 : Time 0 and before last lateral at that time
        - section 2 : from section 1 to end at time 0
        - section 3 : sctrictly after time 0  
    """
    #import pdb; pdb.set_trace()
    # find cut values
    cut12 =  max(df.iloc[0]['coord_y'] for df in lat_dfs.values() if df.iloc[0]['coord_t'] <= 1) \
        if lat_dfs and bool([df.iloc[0]['coord_y'] for df in lat_dfs.values() if df.iloc[0]['coord_t'] <= 1])\
        else None


    cut23 =  mdf[mdf['coord_t'] == 0]['coord_y'].max()
    
    print(f'cut values for sections : {cut12} and {cut23}')

    if cut12: # case where lateral roots exists from the beginning
        # splitting main df in 3 section ?
        mdf1 =  (mdf[ mdf['coord_y'] <= cut12 ]).copy()
        #import pdb; pdb.set_trace()
        mdf2 = (mdf[(mdf['coord_y'] > cut12)  & (mdf['coord_y'] <= cut23)  ]).copy()
        mdf3 = (mdf[ mdf['coord_y'] > cut23 ]).copy()
    else: 
        mdf1 = None
        mdf2 = (mdf[(mdf['coord_y'] <= cut23)  ]).copy()
        mdf3 = (mdf[ mdf['coord_y'] > cut23 ]).copy()

    # splitting lateral roots in 3 packages
    lat_dfs1 = {}
    lat_dfs2 = {}
    lat_dfs3 = {}

    if cut12: # case where lateral roots exists from the beginning
        for (k,df) in lat_dfs.items():
            if not df.empty and df.iloc[0]['coord_y'] <= cut12:
                lat_dfs1.update({k:df.copy()})
            elif not df.empty and df.iloc[0]['coord_y'] > cut12 and  df.iloc[0]['coord_y'] < cut23:
                lat_dfs2.update({k:df.copy()})
            else:
                lat_dfs3.update({k:df.copy()})
    else: 
        for (k,df) in lat_dfs.items():
            if not df.empty and df.iloc[0]['coord_y'] < cut23:
                lat_dfs2.update({k:df.copy()})
            else:
                lat_dfs3.update({k:df.copy()})

    return mdf1, lat_dfs1, mdf2, lat_dfs2, mdf3, lat_dfs3

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
    use_sections = kwargs['use_sections']
    
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
    mdf1 =  dfs['1.1'] 
    mdf2 = dfs['2.1']
    lat_dfs1 = { k:dfs[k] for k in dfs if k.startswith('1') and k!='1.1' }
    lat_dfs2 = { k:dfs[k] for k in dfs if k.startswith('2') and k!='2.1' }
    luj1, nb_lat1 = compute_mainplantroot_len_until_last_junction(
        mdf1, 
        lat_dfs1, 
        scale_factor=scale_factor
    )
    luj2, nb_lat2 = compute_mainplantroot_len_until_last_junction(
        mdf2,
        lat_dfs2, 
        scale_factor=scale_factor
        )
    
    mdf1.insert(7, COLUMNS[2], luj1 , True )
    mdf1.insert(8, COLUMNS[3], nb_lat1 , True )
    mdf2.insert(7, COLUMNS[2], luj2 , True )
    mdf2.insert(8, COLUMNS[3], nb_lat2 , True )

    dfs.update({'1.1':mdf1, '2.1':mdf2 })
    #import pdb; pdb.set_trace()


    
    
    
    # gather data per time
    dfs = {k:unique_merge_per_column(dfs[k], keep_last_point_data_only= True) for k in dfs}
    mdf1 =  dfs['1.1'] 
    mdf2 = dfs['2.1']
    lat_dfs1 = { k:dfs[k] for k in dfs if k.startswith('1') and k!='1.1' }
    lat_dfs2 = { k:dfs[k] for k in dfs if k.startswith('2') and k!='2.1' }
    
    # remove fractional frames if necessary
    if remove_fractional_frames:
        dfs = {k:remove_fractional_frames_in_df(dfs[k]) for k in dfs}

   
    # generate dataframe of section
    
    if use_sections:
        
        mdf11, lat_dfs11, mdf12, lat_dfs12, mdf13, lat_dfs13= gather_dataframes_for_sections(mdf1, lat_dfs1)
        mdf21, lat_dfs21, mdf22, lat_dfs22, mdf23, lat_dfs23 = gather_dataframes_for_sections(mdf2, lat_dfs2)
        

        

    # gather subroots of plantroot 1 and 2
    dfs1 =  {k:dfs[k] for k in dfs if k.startswith('1')  }
    dfs2 =  {k:dfs[k] for k in dfs if k.startswith('2')  }


    ## Now end of column computation : 
    # remove unecessary data from dataframes 
    
    dfs.update( {k:dfs[k].drop(columns = columns_kept_only_once ) for k in list(dfs.keys())[1:]} )
    dfs = { k:dfs[k].drop(columns = columns_to_drop ) for k in dfs }
    dfs1.update( {k:dfs1[k].drop(columns = columns_kept_only_once ) for k in list(dfs1.keys())[1:]} )
    dfs1 = { k:dfs1[k].drop(columns = columns_to_drop ) for k in dfs1 }
    dfs2.update( {k:dfs2[k].drop(columns = columns_kept_only_once ) for k in list(dfs2.keys())[1:]} )
    dfs2 = { k:dfs2[k].drop(columns = columns_to_drop ) for k in dfs2 }
    if use_sections:
        lat_dfs11 = { k:v.drop(columns = columns_to_drop ) for (k,v) in lat_dfs11.items()} 
        lat_dfs12 = { k:v.drop(columns = columns_to_drop ) for (k,v) in lat_dfs12.items()}
        lat_dfs13 = { k:v.drop(columns = columns_to_drop ) for (k,v) in lat_dfs13.items()}
        lat_dfs21 = { k:v.drop(columns = columns_to_drop ) for (k,v) in lat_dfs21.items()} 
        lat_dfs22 = { k:v.drop(columns = columns_to_drop ) for (k,v) in lat_dfs22.items()}
        lat_dfs23 = { k:v.drop(columns = columns_to_drop ) for (k,v) in lat_dfs23.items()}
    
    # add prefix to columns
    dfs = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in dfs }
    dfs1 = {k:add_prefix_except_column(dfs1[k], k + ' ', column_to_exclude= column_merge ) for k in dfs1 }
    dfs2 = {k:add_prefix_except_column(dfs2[k], k + ' ', column_to_exclude= column_merge ) for k in dfs2 }

    if use_sections:
        lat_dfs11 = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in lat_dfs11 } 
        lat_dfs12 = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in lat_dfs12 }
        lat_dfs13 = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in lat_dfs13 }
        lat_dfs21 = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in lat_dfs21 } 
        lat_dfs22 = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in lat_dfs22 }
        lat_dfs23 = {k:add_prefix_except_column(dfs[k], k + ' ', column_to_exclude= column_merge ) for k in lat_dfs23 }


    # merge dfs     
    df1 = merge_list_of_plantroot_dfs(list(dfs1.values()), column = column_merge)
    df2 = merge_list_of_plantroot_dfs(list(dfs2.values()), column = column_merge)
    df = merge_list_of_plantroot_dfs(list(dfs.values()), column = column_merge)
 
    if use_sections:
        df11 = merge_list_of_plantroot_dfs(list(lat_dfs11.values()), column = column_merge) if lat_dfs11 else None
        #import pdb; pdb.set_trace()
        df12 = merge_list_of_plantroot_dfs(list(lat_dfs12.values()), column = column_merge) if lat_dfs12 else None
        df13 = merge_list_of_plantroot_dfs(list(lat_dfs13.values()), column = column_merge) if lat_dfs13 else None
        
        df21 = merge_list_of_plantroot_dfs(list(lat_dfs21.values()), column = column_merge) if lat_dfs21 else None
        df22 = merge_list_of_plantroot_dfs(list(lat_dfs22.values()), column = column_merge) if lat_dfs22 else None
        df23 = merge_list_of_plantroot_dfs(list(lat_dfs23.values()), column = column_merge) if lat_dfs23 else None


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

    
    if use_sections:
        subfiles_folder_path.mkdir(exist_ok=True, parents=True)
        if df11 is not None and not df11.empty: 
            df11.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root1_section1.csv'), index=False)
        if df12 is not None and not df12.empty:    
            df12.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root1_section2.csv'), index=False)
        if df13 is not None and not df13.empty:
            df13.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root1_section3.csv'), index=False)
        if df21 is not None and not df21.empty:
            df21.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root2_section1.csv'), index=False)
        if df22 is not None and not df22.empty:
            df22.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root2_section2.csv'), index=False)
        if df23 is not None and not df23.empty:
            df23.to_csv(Path(subfiles_folder_path,  input_file_path.stem + '_root2_section3.csv'), index=False)
   



def main(*args, **kwargs):
    for arg in args: 
        _main(arg, **kwargs)



if __name__=='__main__':
    #parsing
    # ex 1 : python rsml_parser.py p04.rsml --save-subfiles --remove-fractional-frames --use-sections 
    # ex 2 : python rsml_parser.py 6.rsml --save-subfiles --remove-fractional-frames --use-sections 
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


    parser.add_argument('--use-sections', 
                        action='store_true',
                        help='for a given time frame, keep only last point data'
    )

    parser.add_argument('--one-type-of-column-per-csv', 
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
        use_sections = args.use_sections
    )

    print('end of parsing and conversion')




