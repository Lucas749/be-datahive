"""
Data processing code for BE-dataHIVE: a Base Editing Database
Author: Lucas Schneider (lucas.schneider@cs.ox.ac.uk)
Studies:
    Marquart
        authors                 Marquart et al.
        title                   Predicting base editing outcomes with an attention-based deep learning algorithm trained on high-throughput target library screens
        available under         https://www.nature.com/articles/s41467-021-25375-z    
    Arbab
        authors                 Arbab et al.
        title                   Determinants of Base Editing Outcomes from Target Library Analysis and Machine Learning
        available under         https://www.sciencedirect.com/science/article/pii/S0092867420306322
    Yuan
        authors                 Yuan et al.
        title                   Optimization of C-to-G base editors with sequence context preference predictable by machine learning methods
        available under         https://www.nature.com/articles/s41467-021-25217-y 
    Song
        authors                 Song et al.
        title                   Sequence-specific prediction of the efficiencies of adenine and cytosine base editors
        available under         https://www.nature.com/articles/s41587-020-0573-5
    Pallaseni
        authors                 Pallaseni et al.
        title                   Predicting base editing outcomes using position-specific sequence determinants
        available under         https://academic.oup.com/nar/article/50/6/3551/6548303?login=false

Folder Structure:
    run_processing.py             - runs all data processing functions
    initial_data_processing.py    - contains all functions for the initial data processing of study data
        |__ parse_marquart_data             - parse marquart study data and process it
        |__ flatten_arbab_data              - flatten arbab study data from multiple pickle files to one csv
        |__ process_arbab_data              - process arbab data and calculate bystander views
        |__ parse_pallaseni_data            - parse pallaseni study data and process it
        |__ parse_yuan_data                 - parse yuan study data and process it
        |__ parse_song_data                 - parse song study data and process it
    data_enrichment.py            - contains all functions used to enrich base editing data
        |__ run_cas_offinder                - run cas-offinder to find matching locations on assembly
        |__ add_cas_offinder_results        - add cas-offinder results to data files
        |__ match_location_via_biopython    - match sequence location via biopython - only used for arbab as chromosome and location is approximately known
        |__ get_encode_data                 - get encode screen data
        |__ run_RNA_fold                    - run RNAfold to calculate MFE energy term
        |__ unpack_RNA_fold_output          - unpack RNA fold output into a data frame
        |__ run_CRISPR_spec                 - run CRISPRspec to calculate free energy terms
        |__ calculate_melting_temperatures  - calculate melting temperatures
        |__ harmonize_data_padding          - Harmonize data and add padding to the target sequence
        |__ add_pam_col_grna_seq_mismatch   - add PAM columns and grna/seq mismatch flag
        |__ add_crispr_spec_energies        - add CRISPR spec energies to data file
        |__ add_screen_data                 - add SCREEN data to data file
        |__ add_RNAfold_energy              - add RNAfold MFE to data file
        |__ add_melting_temperature         - add melting temperature to data files
        |__ merge_data_files                - merge all author data files into one file
        |__ get_study_data_stats            - calculate stats for individual studies
        |__ get_data_stats                  - calculate stats for the final data file
    encodings.py                    - contains all functions used to encode base editing data
        |__ one_hot_encode_seq              - returns one-hot encoding for DNA sequence
        |__ one_hot_encode_dataframe        - encodes data in selected columns via one-hot encoding
        |__ hindex_to_xy                    - returns the x and y coordinates in a 2D space corresponding to the given index, as per the Hilbert curve ordering
        |__ write_pixel_list_hilbert        - returns a list of points in the 2D space according to the Hilbert curve ordering
        |__ make_image                      - generate 32x32 image mapping
        |__ hilbert_curve_encode_dataframe  - encodes data in selected columns via hilbert curve encoding
"""
#Import packages
import io
import numpy as np
import pandas as pd

#####################################################################################################################################################################
# Encoding
#####################################################################################################################################################################
#######################################################
# One-hot
#######################################################
#Returns one-hot encoding for DNA sequence
def one_hot_encode_seq(seq):
    """
    Returns one-hot encoding for DNA sequence
    @params:
        seq                              - Required  : DNA sequence (str)
    """
    # Mapping for each base to its one-hot encoding
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1], 'N':[0,0,0,0]}
    
    # Initialize an empty list to store the encoding
    encoding = []
    
    # Iterate over the sequence
    for base in seq:
        # Add the one-hot encoding for the base to the list
        encoding.append(mapping[base])
        
    # Convert the list to a NumPy array and transpose it
    return np.array(encoding).T


#Encodes data in selected columns via one-hot encoding
def one_hot_encode_dataframe(bystander, cols_to_encode):
    """
    Encodes data in selected columns via one-hot encoding
    @params:
        bystander                        - Required  : data frame containing the genetic data (pd.Dataframe)
        cols_to_encode                   - Required  : columns to encode (list)
    """
    # Iterate over the columns to encode
    for column in cols_to_encode:

        encoded_list = []
        for val in bystander[column]:
            try:
                encoding = one_hot_encode_seq(val)

                #Convert to byte string                
                buf1 = io.BytesIO()
                np.save(buf1, encoding)
                encoded_list.append(buf1.getvalue().decode('latin-1'))
            except:
                encoded_list.append([])

        temp_df = pd.DataFrame({f"one_hot_{column}": encoded_list})

        bystander = pd.concat([bystander,temp_df],axis=1)

    return bystander

#######################################################
# Hilbert Curve
# Code based on: https://github.com/iatahmid/chilenpred
#######################################################
#Returns the x and y coordinates in a 2D space corresponding to the given index, as per the Hilbert curve ordering
def hindex_to_xy(hindex, N):
    """
    Returns the x and y coordinates in a 2D space corresponding to the given index, as per the Hilbert curve ordering
    @params:
        hindex                       - Required  : hilbert index (int)
        N                            - Required  : size of grid (int)
    """
    positions = [[0, 0], [0, 1], [1, 1], [1, 0]]
    tmp = positions[hindex & 3]
    hindex = hindex >> 2
    x = tmp[0]
    y = tmp[1]
    n = 4

    while (n <= N):
        n2 = int(n / 2)
        pos_in_small_square = hindex & 3
        if(pos_in_small_square == 0): # lower left
            x, y = y, x
        elif(pos_in_small_square == 1): # upper left
            y = y + n2
        elif(pos_in_small_square == 2): # upper right
            x, y = x + n2, y + n2
        elif(pos_in_small_square == 3): # lower right
            x, y = (n2 - 1) - y + n2, (n2 - 1) - x
        hindex = hindex >> 2
        n *= 2

    return x, y

#Returns a list of points in the 2D space according to the Hilbert curve ordering
def write_pixel_list_hilbert(order):
    """
    Returns a list of points in the 2D space according to the Hilbert curve ordering
    @params:
        order                            - Required  : order for hilbert curve (int)
    """
    point_list = []
    N = 2**order;
    for i in range(N*N):
        curr = hindex_to_xy(i, N)
        point_list.append(curr)		
    
    return point_list

#Generate 32x32 image mapping
def make_image(sequence, mapping, point_list):
    """
    Generate 32x32 image mapping
    @params:
        sequence                           - Required  : DNA sequence (str)
        mapping                            - Required  : sequence char mapping (dict)
        point_list                         - Required  : hilbert curve point space (list)
    """
    image_array = np.ones((32,32,4))
    for k in range(len(sequence)):
        x,y = point_list[k]
        image_array[x][y] = mapping[sequence[k].upper()]
    
    return image_array

#Encodes data in selected columns via hilbert curve encoding
def hilbert_curve_encode_dataframe(bystander, cols_to_encode, order = 5):
    """
    Encodes data in selected columns via hilbert curve encoding
    @params:
        bystander                        - Required  : data frame containing the genetic data (pd.Dataframe)
        cols_to_encode                   - Required  : columns to encode (list)
        order                            - Required  : order for hilbert curve (int)
    """
    #Set point list
    point_list = write_pixel_list_hilbert(order)

    mapping = {'A': np.array([1,0,0,0]), 'T': np.array([0,1,0,0]), 
            'C': np.array([0,0,1,0]), 'G': np.array([0,0,0,1]), 
            'N': np.array([0,0,0,0])}
    
    # Iterate over the columns to encode
    for column in cols_to_encode:
        # Create one-hot encodings

        val = bystander[column][0]

        encoded_list = []
        for val in bystander[column]:
            try:
                encoding = make_image(val, mapping, point_list)
            
                buf1 = io.BytesIO()
                np.save(buf1, encoding)
                encoded_list.append(buf1.getvalue().decode('latin-1'))
            except:
                encoded_list.append([])

        temp_df = pd.DataFrame({f"hilbert_curve_{column}": encoded_list})

        bystander = pd.concat([bystander,temp_df],axis=1)

    return bystander
    