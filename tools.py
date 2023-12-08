import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def load_finfo_probe(
    path,
    extract_platform_quadrants=False
    ):
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
            if file.endswith('.xlsx') and "COORDINATES" in file and extract_platform_quadrants:
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
                    "phase": f_comps[2]
                }
                if extract_platform_quadrants:
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
