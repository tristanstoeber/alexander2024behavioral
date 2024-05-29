import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def detect_center_based_on_quadrant_coordinates(q1, q2, q3, q4):
    # Extract coordinates from the tuples
    q1_x, q1_y = q1
    q2_x, q2_y = q2
    q3_x, q3_y = q3
    q4_x, q4_y = q4

    # Check for coincident points which might lead to incorrect calculations
    if q1 == q3 or q2 == q4:
        return None

    # Check for vertical lines and handle them separately
    is_q1q3_vertical = q1_x == q3_x
    is_q2q4_vertical = q2_x == q4_x

    if is_q1q3_vertical and is_q2q4_vertical:
        return None  # Parallel vertical lines do not intersect

    if is_q1q3_vertical:
        # Calculate the intersection point using the non-vertical line's equation
        # The x-coordinate is that of the vertical line, and y is calculated from the other line's equation
        m2 = (q4_y - q2_y) / (q4_x - q2_x)  # slope of q2-q4 line
        c2 = q2_y - m2 * q2_x              # intercept of q2-q4 line
        return q1_x, m2 * q1_x + c2

    if is_q2q4_vertical:
        # Similarly, calculate for the other vertical line
        m1 = (q3_y - q1_y) / (q3_x - q1_x)  # slope of q1-q3 line
        c1 = q1_y - m1 * q1_x              # intercept of q1-q3 line
        return q2_x, m1 * q2_x + c1

    # Calculate slopes and intercepts for non-vertical lines
    m1 = (q3_y - q1_y) / (q3_x - q1_x)
    c1 = q1_y - m1 * q1_x

    m2 = (q4_y - q2_y) / (q4_x - q2_x)
    c2 = q2_y - m2 * q2_x

    # Check if lines are parallel (slopes are equal)
    if m1 == m2:
        return None  # No intersection for parallel lines

    # Find intersection point for non-vertical, non-parallel lines
    center_x = (c2 - c1) / (m1 - m2)
    center_y = m1 * center_x + c1

    return center_x, center_y


def extract_radius_from_quadrant_positions(q1, q2, q3, q4):
    Q = [q1, q2, q3, q4]
    
    ls_diameter = []
    for i in np.arange(2):
        vec_q = np.array(Q[i])-np.array(Q[2+i])
        diameter_i = np.sqrt(np.sum(vec_q**2))
        ls_diameter.append(diameter_i)
    radius = np.mean(ls_diameter)/2.
    
    return radius

def detect_circle_boundaries(x_coordinates, y_coordinates):
    # Find the maximum and minimum x, y coordinates
    max_x, min_x = max(x_coordinates), min(x_coordinates)
    max_y, min_y = max(y_coordinates), min(y_coordinates)

    # Calculate the center coordinates as the midpoint between max and min values
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    # Calculate the radius as half of the maximum distance between x or y values
    radius = max(max_x - min_x, max_y - min_y) / 2

    return center_x, center_y, radius

def load_finfo_train(path):
    ls_finfo = []
    
    # Walk through the directory to extract trial info
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check for .xlsx files that do not contain 'COORDINATES' in their name
            if file.endswith('.xlsx') and "Coordinates" not in file:
                # Construct the relative path of the file
                relative_path = os.path.join(root, file)
    
                # Split the filename into components
                f_comps = file.split('.')
    
                # Append file information to the list
                dct_i = {
                    "fname": file,
                    "relative_path": relative_path,
                    "experiment": f_comps[0],
                    "cohort": f_comps[1],
                    "phase": f_comps[2],
                    "animal_id": f_comps[3],
                    "day": int(f_comps[4][1:]),
                    "trial": int(f_comps[5][1:]),
                }
                ls_finfo.append(dct_i)

    # Create a DataFrame from the list of file information
    df_finfo = pd.DataFrame(ls_finfo)

    # do another walkthrough to extract maze info
    ls_mazeinfo = []
    for root, dirs, files in os.walk(path):

        # extract platform and quadrant info
        for file in files:        
            if file.endswith('.xlsx') and (" Coordinates" in file or ".Coordinates" in file):
                # get coordinates
                path_i = os.path.join(root, file)
                df = pd.read_excel(path_i, usecols=[0, 1, 2], index_col=0, header=[2])
                df['xy'] = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
                
                dct_info = {}
                
                info_platform = df.loc[df.index.str.contains('Platform'), 'xy']
                platform_name = info_platform.index[0]
                platform_pos = info_platform.values[0]
                dct_mazeinfo = {
                    'platform_name': platform_name,
                    'platform_pos': platform_pos
                }               
                # add quadrant info
                info_quadrats = df[df.index.str.startswith('Q')]['xy'].to_dict()
                dct_mazeinfo.update(info_quadrats)

                # Split the filename into components
                f_comps = file.split('.')
    
                # Append file information to the list
                dct_i = {
                    "experiment": f_comps[0],
                    "cohort": f_comps[1],
                    "phase": f_comps[2],
                    "day": int(f_comps[3].split()[0][3:]),
                }
                dct_mazeinfo.update(dct_i)

                ls_mazeinfo.append(dct_mazeinfo)
    df_mazeinfo = pd.DataFrame(ls_mazeinfo)
    df_finfo = pd.merge(df_finfo, df_mazeinfo, on=['experiment', 'cohort', 'phase', 'day'], how='left')
    return df_finfo

def load_finfo_probe(
    path):
    """
    Scans a directory and its subdirectories for .xlsx files that do not contain 'COORDINATES' in their filename.
    Extracts information from each qualifying file's name and path, and compiles it into a pandas DataFrame.

    Parameters:
    path (str): The path to the directory where the .xlsx files are stored.

    Returns:
    pd.DataFrame: A DataFrame containing information extracted from the file names and paths.

    The DataFrame contains the following columns:
    - 'fname': The name of the file.
    - 'relative_path': The relative path of the file.
    - 'experiment': The first component of the file name, presumed to be the experiment name.
    - 'animal_id': The second component of the file name, presumed to be the animal ID.
    - 'phase': The third component of the file name, presumed to be the phase of the experiment.
    """
    
    # Initialize a list to store file information
    ls_finfo = []

    # Walk through the directory
    for root, dirs, files in os.walk(path):

        # extract platform and quadrant info
        for file in files:        
            if file.endswith('.xlsx') and "COORDINATES" in file:
                # get coordinates
                path_i = os.path.join(root, file)
                df = pd.read_excel(path_i, usecols=[0, 1, 2, 3], index_col=0, header=[0])
                df['xy'] = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
                df['Marked'] = df['Marked'].astype(str)
                
                dct_info = {}
                
                info_platform = df.loc[(df['Marked'] == 'x') & df.index.str.contains('Platform'), 'xy']
                platform_name = info_platform.index[0]
                platform_pos = info_platform.values[0]

                dct_pltfrm_quadrants = {
                    'platform_name': platform_name,
                    'platform_pos': platform_pos
                }               
                
                # add quadrant info
                info_quadrats = df[df.index.str.startswith('Q')]['xy'].to_dict()
                dct_pltfrm_quadrants.update(info_quadrats)
                
        for file in files:
            # Check for .xlsx files that do not contain 'COORDINATES' in their name
            if file.endswith('.xlsx') and "COORDINATES" not in file:
                # Construct the relative path of the file
                relative_path = os.path.join(root, file)

                # Split the filename into components
                f_comps = file.split('.')

                # Append file information to the list
                dct_i = {
                    "fname": file,
                    "relative_path": relative_path,
                    "experiment": f_comps[0],
                    "animal_id": f_comps[1],
                    "phase": f_comps[2],
                    "trial": None,
                    "cohort": f_comps[1][0],
                }
                dct_i.update(dct_pltfrm_quadrants)
                ls_finfo.append(dct_i)

    # Create a DataFrame from the list of file information
    df_finfo = pd.DataFrame(ls_finfo)

    return df_finfo

def load_recording(
    rel_path,
    header=[38, 39],
    column_sel=[
        'Trial time',
        'Recording time',
        'X center',
        'Y center']):
    """
    Loads and processes an Excel file based on the specified parameters.

    Parameters:
    rel_path (str): The relative path to the Excel file.
    header (list, optional): The rows to use as the header (default is [38, 39]).
    column_sel (list, optional): The specific columns to select from the file.

    Returns:
    pd.DataFrame: A DataFrame containing the selected columns from the Excel file.

    This function reads an Excel file, using the specified header rows for column names,
    and then selects a subset of columns based on the column_sel parameter.
    """
    
    # Read the Excel file using the specified header rows
    df = pd.read_excel(rel_path, header=header)

    # Select the specified columns from the DataFrame
    df = df[column_sel]

    return df

def determine_start_end_of_blobs(a, min_len=None):
    m = np.concatenate(( [True], ~a, [True] ))  # Mask                                                                                                                                                             
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits                                                                                                                                       
    if min_len:
        ss_bool = (ss[:,1] - ss[:,0]) >= min_len
        ss = ss[ss_bool]
        # if no value is present modify output                                                                                                                                                                     
        if ~np.any(ss_bool):
            ss = ss.flatten()
    return ss

def time_in_blobs(ss, t):
    dt = np.median(np.diff(t))
    # add one timestep
    t = np.append(t, np.max(t)+dt)
    ls_t = []
    for ss_i in ss:
        dt = t[ss_i[1]] - t[ss_i[0]]
        ls_t.append(dt)
    return np.sum(ls_t)
