"""
Data processing code for BE-dataHIVE: a MySQL Base Editing Database for Practitioners and Computer Scientists
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
"""
#Import packages
import pandas as pd
import os
import numpy as np

#####################################################################################################################################################################
# Initial Data Processing
#####################################################################################################################################################################
#######################################################
# Marquart
#######################################################
# Parse marquart study data and process it
def parse_marquart_data(file_name, editing_window_start, editing_window_end, num_positions_before_protospacer, num_positions, columns, file_path, saving_path, file_path_ext):
    """
    Parse marquart study data and process it
    @params:
        file_name                            - Required  : file name of the data file (str)
        editing_window_start                 - Required  : start base position of the editing window (int)
        editing_window_end                   - Required  : end base position of the editing window (int)
        num_positions_before_protospacer     - Required  : number of unique base positions before protospacer index (int)
        num_positions                        - Required  : number of unique base positions after protospacer index (int)
        columns                              - Required  : column names for the outcome data frame (list)
        file_path                            - Required  : file path to the raw data file (str)
        saving_path                          - Required  : file path to save the processed data file (str)
        file_path_ext                        - Required  : file path for folder route (str)
    """
    # Load data file
    os.chdir(file_path + file_path_ext)
    xl_sheet_names = pd.ExcelFile(file_name).sheet_names
    sheet_names = [n for n in xl_sheet_names if n.endswith('proportions')]

    # Generate position column names for position outcome edits
    pos_columns = [f"Position_{i}" for i in range(num_positions_before_protospacer, num_positions+1)]
    pos_columns = [[f"{i} A", f"{i} T", f"{i} C", f"{i} G"]
                   for i in pos_columns]
    pos_columns = [x for xs in pos_columns for x in xs]
    columns_df = columns + pos_columns

    # Create empty df for postion outcome edits
    df = pd.DataFrame(columns=columns_df)

    # Generate position column names for position edits
    pos_columns = [f"Position_{i}" for i in range(num_positions_before_protospacer, num_positions+1)]
    columns_df = columns + pos_columns

    # Create empty df for postion edits
    df_pos_edits = pd.DataFrame(columns=columns_df)

    # Generate position index frames to calculate editing efficiency
    pos_indexes = range(0, num_positions+1)
    zero_pos = np.where(np.array(list(pos_indexes)) == 0)[0][0]
    edit_window_pos_start = zero_pos + editing_window_start
    edit_window_pos_end = zero_pos + editing_window_end

    # Loop through all data sheets
    for sheet_name in sheet_names:
        data_by = pd.read_excel(file_name, sheet_name=sheet_name)
        base_editor = sheet_name.split("_")[0]
        data_by.index = data_by.ID + '_' + base_editor

        # Unpack every data point
        for i, ind in enumerate(data_by.index):
            print(f"Sheet name: {sheet_name} - Unpacking {i} / {len(data_by)}")

            #Fixed value for pam index
            pam_index = 20

            #Unpack data
            temp_data = data_by.loc[[ind], :]
            protospace_position = 0
            total_count = temp_data.Counts[0]
            or_seq = temp_data.Reference[0]
            target = or_seq

            # Generate outcomes df
            outcomes = [list(out) for out in temp_data.Outcome]
            outcomes = pd.DataFrame(outcomes)

            fraction_edited = temp_data.Proportion
            efficiency = fraction_edited[temp_data.Outcome != target].sum()
            edited_count = int(
                fraction_edited[temp_data.Outcome == target].sum() * total_count)
            scaling_factor_edited_count_denominator = total_count / edited_count

            # Calculate editing percentage
            for k, nt in enumerate(list(or_seq)):
                pos_index = pos_indexes[k]
                all_nt_pos = outcomes.iloc[:, k].unique()
                for nt_pos in all_nt_pos:
                    df.loc[ind, f"Position_{pos_index} {nt_pos}"] = fraction_edited[outcomes.iloc[:, k].values == nt_pos].sum(
                    )

                # Add view editing percentage at position
                df_pos_edits.loc[ind, f"Position_{pos_index}"] = fraction_edited[outcomes.iloc[:, k].values != nt].sum(
                )

            # Calculate editing effiency
            editing_window_efficiency = fraction_edited[np.any(outcomes.iloc[:, edit_window_pos_start:edit_window_pos_end].values != list(
                or_seq[edit_window_pos_start:edit_window_pos_end]), axis=1)].sum()

            # Assign values - fixed values are based on the study
            grna = or_seq
            sequence = or_seq
            original_id = temp_data.ID[0]
            genomic_context = "NA"
            genomic_context_build = "NA"
            cell = "HEK293T"

            df.loc[ind, ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'scaling_factor_edited_count_denominator',
                         f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated']] = [original_id, grna, or_seq, sequence, protospace_position, pam_index, genomic_context, genomic_context_build, cell, base_editor, total_count, edited_count, efficiency, scaling_factor_edited_count_denominator, editing_window_efficiency]

    # Add repeating data to dfs
    df_pos_edits.loc[:, ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated']] = df.loc[:, [
        'original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated']]

    # Save data
    os.chdir(saving_path + file_path_ext)
    df.to_csv('bystander_outcome_per_position.csv')
    df_pos_edits.to_csv('bystander_edit_per_position.csv')

#######################################################
# Arbab
#######################################################
# Flatten arbab study data from multiple pickle files to one csv
def flatten_arbab_data(columns, file_path, eff_data_path, saving_path, file_path_ext):
    """
    Flatten arbab study data from multiple pickle files to one csv
    @params:
        columns                      - Required  : column names for the outcome data frame (list)
        file_path                    - Required  : file path to the raw data file (str)
        eff_data_path                - Required  : file path to the raw effiency data (str)
        saving_path                  - Required  : file path to save the processed data file (str)
        file_path_ext                - Required  : file path for folder route (str)
    """
    #Set directory
    os.chdir(file_path + file_path_ext)
    
    #Get all file names
    file_names = os.listdir()

    #Create empty data frame
    df = pd.DataFrame(columns=columns)

    # Concat outcomes out of pkl to csv flat format
    for r, file_name in enumerate(file_names):

        # Load data csv for additonal information
        print(
            f"Loading file {file_name} - {r} / {len(file_names)} | length df {len(df)}")
        
        #Try reading efficiency data for the corresponding bystander data
        try:
            eff_data_path_file = eff_data_path + \
                '\\' + file_name.split('.')[0] + '.csv'
            eff_data = pd.read_csv(eff_data_path_file, index_col=0)
            eff_data.index = eff_data.loc[:, 'Name (unique)']
        except Exception as e:
            print(f"Error: {e} - skipping file")
            continue

        #Extract information from file name
        file_format = file_name.split('_')[1]
        cell = file_name.split('_')[0]
        base_editor = file_name.split('_')[-1].split('.')[0]

        #Load pickle data with the bystander data
        data_by = pd.read_pickle(file_name)

        #Unpack bystander data
        for t, key in enumerate(list(data_by.keys())):
            #Init duplicated key to 0 - tracks if keys occur multiple times
            duplicated_key = 0
            print(f"{t} / {len(list(data_by.keys()))}")

            try:
                # Get bystander data
                data_temp = data_by[key]
                exp_edited_count = data_temp.loc[:, 'Count'].sum()

                #Init error flag - tracks if an error occurs during handling
                error_flag = 0

                #Routine if keys are duplicated
                if sum(eff_data.index == key) > 1:
                    duplicated_key = 1

                    #Get all efficiency data
                    temp_eff_data = eff_data.loc[key, :]

                    # Filter out empty columns for the key
                    subset_mask = np.any(
                        pd.isnull(temp_eff_data.loc[key, :].values) == False, axis=0)
                    col_with_values = eff_data.columns[subset_mask]

                    #Unwrap data
                    grna = temp_eff_data.loc[key, 'gRNA (20nt)'][0]
                    if file_format == "CtoGA":
                        sequence = temp_eff_data.loc[key,
                                                    'Sequence context (35 or 61 nt)'][0]
                    else:
                        sequence = temp_eff_data.loc[key,
                                                    'Sequence context (56nt)'][0]

                    #Get genomic context 
                    try:
                        if file_format == "CtoT" or file_format == "AtoG" or file_format == "CtoGA":
                            chrom_field = [c for c in col_with_values if c.startswith(
                                'hg19 chromosome')][0]
                            coordinate_field = [
                                c for c in col_with_values if c.startswith('hg19 coordinate')][0]
                            strand_field = [
                                c for c in col_with_values if c.startswith('hg19 strand')][0]

                            genomic_context = f"chr{temp_eff_data.loc[key,chrom_field][0]}:{int(temp_eff_data.loc[key,coordinate_field][0])}_{temp_eff_data.loc[key,strand_field][0]}"
                        else:
                            cont_field = [c for c in col_with_values if c.startswith(
                                'Genomic context') and not c.endswith('build')][0]

                            temp_gen_cont = temp_eff_data.loc[key, cont_field][0]
                            genomic_context = f"{temp_gen_cont.split('_')[0][1:]}:{temp_gen_cont.split('_')[2]}-{temp_gen_cont.split('_')[3]}"
                    except Exception as e:
                        print("ERROR Genomic Context: {e}")
                        genomic_context = 'NA'

                    #Get genomic context build
                    try:
                        if file_format == "CtoT" or file_format == "AtoG" or file_format == "CtoGA":
                            genomic_context_build = "hg19"
                        else:
                            cont_field = [c for c in col_with_values if c.startswith(
                                'Genomic context build')][0]

                            genomic_context_build = temp_eff_data.loc[key,
                                                                    cont_field][0]

                    except Exception as e:
                        print("ERROR Genomic Context Build: {e}")
                        genomic_context_build = 'NA'

                    protospace_position = sequence.find(grna)
                    pam_index = protospace_position + 20

                    # Filter out empty columns for the key
                    try:
                        frac_field = [c for c in col_with_values if c.startswith(
                            'Fraction edited')][0]
                        eff_edited_perc = temp_eff_data.loc[key, frac_field]
                    except:
                        eff_edited_perc = "NA"

                    total_field = [c for c in col_with_values if c.startswith(
                        'Total count')]  # [0]
                    edited_field = [c for c in col_with_values if c.startswith(
                        'Edited count')]
                    total_count = temp_eff_data.loc[key, total_field].sum().sum()
                    edited_count = temp_eff_data.loc[key, edited_field].sum().sum()

                    total_count_aggregated = eff_data.loc[key, total_field].sum(
                    ).sum()

                else:
                    # Get columns with values
                    col_with_values = eff_data.columns[pd.isnull(
                        eff_data.loc[key, :].values) == False]

                    grna = eff_data.loc[key, 'gRNA (20nt)']

                    if file_format == "CtoGA":
                        sequence = eff_data.loc[key,
                                                'Sequence context (35 or 61 nt)']
                    else:
                        sequence = eff_data.loc[key, 'Sequence context (56nt)']

                    #Get genomic context
                    try:
                        if file_format == "CtoT" or file_format == "AtoG" or file_format == "CtoGA":
                            chrom_field = [c for c in col_with_values if c.startswith(
                                'hg19 chromosome')][0]
                            coordinate_field = [
                                c for c in col_with_values if c.startswith('hg19 coordinate')][0]
                            strand_field = [
                                c for c in col_with_values if c.startswith('hg19 strand')][0]

                            genomic_context = f"chr{eff_data.loc[key,chrom_field]}:{int(eff_data.loc[key,coordinate_field])}_{eff_data.loc[key,strand_field]}"
                        else:
                            cont_field = [c for c in col_with_values if c.startswith(
                                'Genomic context') and not c.endswith('build')][0]

                            temp_gen_cont = eff_data.loc[key, cont_field]
                            genomic_context = f"{temp_gen_cont.split('_')[0][1:]}:{temp_gen_cont.split('_')[2]}-{temp_gen_cont.split('_')[3]}"
                    except:
                        genomic_context = 'NA'

                    #Get genomic context build
                    try:
                        if file_format == "CtoT" or file_format == "AtoG" or file_format == "CtoGA":
                            genomic_context_build = "hg19"
                        else:
                            cont_field = [c for c in col_with_values if c.startswith(
                                'Genomic context build')][0]

                            genomic_context_build = eff_data.loc[key, cont_field]

                    except:
                        genomic_context_build = 'NA'

                    protospace_position = sequence.find(grna)
                    pam_index = protospace_position + 20 #standard 20 size

                    # Filter out empty columns for the key
                    try:
                        frac_field = [c for c in col_with_values if c.startswith(
                            'Fraction edited')][0]
                        eff_edited_perc = eff_data.loc[key, frac_field]
                    except:
                        eff_edited_perc = "NA"

                    total_field = [
                        c for c in col_with_values if c.startswith('Total count')]
                    edited_field = [
                        c for c in col_with_values if c.startswith('Edited count')]
                    total_count = eff_data.loc[key, total_field].sum()
                    edited_count = eff_data.loc[key, edited_field].sum()

                    total_count_aggregated = total_count

                # Check if edited counts match
                if exp_edited_count != edited_count:
                    error_flag = 1

                row_ind = data_temp.index[0]

                position_change = np.array(
                    [float(c[1:]) for c in data_temp.columns if len(c) == 2 or len(c) == 3])
                position_change = position_change + protospace_position

                # Add prefix to account for duplicates with same id but different data
                if key in df.original_id:
                    duplicate_prefix = "_" + \
                        str(len(
                            [c for c in df.original_id if c.startswith(f"{key}_0")]))
                else:
                    duplicate_prefix = ''

                #Save all data points
                for j, row_ind in enumerate(data_temp.index):
                    temp_seq = list(sequence)

                    nt_values = data_temp.loc[row_ind,
                                            data_temp.columns[:-3]].values
                    for i, p in enumerate(position_change):
                        temp_seq[int(p)] = nt_values[i]

                    outcome = ''.join(temp_seq)

                    fraction_edited_outcome = data_temp.loc[row_ind, 'Frequency']
                    edited_count_outcome = data_temp.loc[row_ind, 'Count']
                    save_ind = f"{key}_{j}" + duplicate_prefix

                    #Assign values
                    df.loc[save_ind, 'original_id'] = key
                    df.loc[save_ind, 'grna'] = grna
                    df.loc[save_ind, 'sequence'] = sequence
                    df.loc[save_ind, 'outcome'] = outcome
                    df.loc[save_ind, 'pam_index'] = pam_index
                    df.loc[save_ind, 'protospace_position'] = protospace_position
                    df.loc[save_ind, 'genomic_context'] = genomic_context
                    df.loc[save_ind, 'genomic_context_build'] = genomic_context_build
                    df.loc[save_ind, 'cell'] = cell
                    df.loc[save_ind, 'base_editor'] = base_editor
                    df.loc[save_ind, 'total_count'] = total_count
                    df.loc[save_ind, 'total_count_aggregated'] = total_count_aggregated
                    df.loc[save_ind, 'edited_count_eff'] = edited_count
                    df.loc[save_ind, 'edited_count_bys'] = exp_edited_count
                    df.loc[save_ind, 'edited_count_outcome'] = edited_count_outcome
                    df.loc[save_ind, 'fraction_edited_outcome'] = fraction_edited_outcome
                    df.loc[save_ind, 'fraction_edited_eff'] = edited_count / total_count
                    df.loc[save_ind, 'fraction_edited_bys'] = exp_edited_count / \
                        total_count_aggregated
                    df.loc[save_ind, 'error_flag'] = error_flag
                    df.loc[save_ind, 'duplicated_key'] = duplicated_key
                    df.loc[save_ind, 'file_name'] = file_name

            except Exception as e:
                print(f"Error: {e}. File {file_name} - key {key}")
                continue

    # Save data
    os.chdir(saving_path + file_path_ext)
    df.to_csv('bystander.csv')

# Process arbab data and calculate bystander views
def process_arbab_data(file_name, editing_window_start, editing_window_end, num_positions_before_protospacer, num_positions, columns, file_path, saving_path, file_path_ext):
    """
    Process arbab data and calculate bystander views
    @params:
        file_name                            - Required  : file name of the data file (str)
        editing_window_start                 - Required  : start base position of the editing window (int)
        editing_window_end                   - Required  : end base position of the editing window (int)
        num_positions_before_protospacer     - Required  : number of unique base positions before protospacer index (int)
        num_positions                        - Required  : number of unique base positions after protospacer index (int)
        columns                              - Required  : column names for the outcome data frame (list)
        file_path                            - Required  : file path to the raw data file (str)
        eff_data_path                        - Required  : file path to the raw effiency data (str)
        saving_path                          - Required  : file path to save the processed data file (str)
        file_path_ext                        - Required  : file path for folder route (str)
    """
    # Edits at positions view
    os.chdir(file_path)
    data_by = pd.read_csv(file_name, index_col=0)

    #Get cell names as they are saved in the file names
    cells_index = []
    for n in data_by.file_name:
        if pd.isnull(n):
            cells_index.append('nan')
        else:
            cells_index.append(n.split('_')[0])

    #Assign unique indices
    data_by.index = data_by.original_id + '_' + \
        cells_index + '_' + data_by.base_editor

    # Generate position column names for position outcome edits
    pos_columns = [f"Position_{i}" for i in range(-num_positions_before_protospacer, num_positions+1)]
    pos_columns = [[f"{i} A", f"{i} T", f"{i} C", f"{i} G"] for i in pos_columns]
    pos_columns = [x for xs in pos_columns for x in xs]
    columns_df = columns + pos_columns
    
    #Create position outcome data frame
    df = pd.DataFrame(columns=columns_df)

    # Generate position column names for position edits
    pos_columns = [f"Position_{i}" for i in range(-num_positions_before_protospacer, num_positions+1)]
    columns_df = columns + pos_columns

    #Create position edits data frame
    df_pos_edits = pd.DataFrame(columns=columns_df)

    # Generate position indices helper
    pos_indexes = range(-num_positions_before_protospacer, num_positions+1)
    zero_pos = np.where(np.array(list(pos_indexes)) == 0)[0][0]
    edit_window_pos_start = zero_pos + editing_window_start
    edit_window_pos_end = zero_pos + editing_window_end

    len_index = len(data_by.index.unique())

    #Loop through all data points
    for i, ind in enumerate(data_by.index.unique()):
        try:
            print(f"Unpacking {i} / {len_index}")

            #Unpack data
            temp_data = data_by.loc[[ind], :]
            protospace_position = int(temp_data.protospace_position[0])
            pam_index = protospace_position + 20
            total_count = temp_data.total_count_aggregated[0]
            edited_count = temp_data.edited_count_bys[0]
            or_seq = temp_data.sequence[0][protospace_position -
                                        10:protospace_position+30]
            outcomes = [list(out[protospace_position - 10:protospace_position+30])
                        for out in temp_data.outcome]
            outcomes = pd.DataFrame(outcomes)

            #Scaling factors to convert ratios to different denominators
            scaling_factor = edited_count / total_count
            scaling_factor_edited_count_denominator = total_count / edited_count

            #Calculate efficiency rate within window
            grna = temp_data.grna[0]
            fraction_edited = temp_data.fraction_edited_outcome

            efficiency = fraction_edited.values[np.any(outcomes.iloc[:, or_seq.find(
                grna):or_seq.find(grna)+len(grna)] != list(grna), axis=1)].sum()

            for k, nt in enumerate(list(or_seq)):
                pos_index = pos_indexes[k]
                # for k in range(outcomes.shape[1]):
                all_nt_pos = outcomes.iloc[:, k].unique()
                for nt_pos in all_nt_pos:
                    df.loc[ind, f"Position_{pos_index} {nt_pos}"] = fraction_edited[outcomes.iloc[:, k].values == nt_pos].sum(
                    ) * scaling_factor

                # Add view editing percentage at position
                df_pos_edits.loc[ind,
                                f"Position_{pos_index}"] = fraction_edited[outcomes.iloc[:, k].values != nt].sum()

            editing_window_efficiency = fraction_edited[np.any(outcomes.iloc[:, edit_window_pos_start:edit_window_pos_end].values != list(
                or_seq[edit_window_pos_start:edit_window_pos_end]), axis=1)].sum()

            #Assign values
            sequence = temp_data.sequence[0]
            genomic_context = temp_data.genomic_context[0]
            genomic_context_build = temp_data.genomic_context_build[0]
            cell = temp_data.cell[0]
            base_editor = temp_data.base_editor[0]
            original_id = temp_data.original_id[0]
            edited_count_eff = temp_data.edited_count_eff[0]
            fraction_edited_eff = temp_data.fraction_edited_eff[0]
            total_count_eff = int(edited_count_eff/fraction_edited_eff)

            total_count_rep = total_count_eff
            edit_rep = edited_count_eff
            efficiency_full_rep = fraction_edited_eff

            #Save data
            df.loc[ind, ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated', 'efficiency_data_total_count', 'efficiency_data_edited_count', 'efficiency_data_efficiency_full_grna', 'efficiency_full_grna_reported', 'total_count_reported_efficiency', 'edited_count_reported_efficiency']] = [
                ind, grna, or_seq, sequence, protospace_position, pam_index, genomic_context, genomic_context_build, cell, base_editor, total_count, edited_count, efficiency, scaling_factor_edited_count_denominator, editing_window_efficiency, total_count_eff, edited_count_eff, fraction_edited_eff, efficiency_full_rep, total_count_rep, edit_rep]
        except Exception as e:
            print(f'ERROR: {e}')

    #Add reported effiencies
    for i, ind in enumerate(df.index):
        #Unwrap data
        be = df.loc[ind, 'base_editor']
        cell = df.loc[ind, 'cell']

        #Bystander data unwrap
        data_sub = data_by.loc[[ind], :].copy()
        data_sub['cell'] = [n.split('_')[0] for n in data_sub.file_name]
        data_sub = data_sub.loc[(data_sub.base_editor == be)
                                & (data_sub.cell == cell), ]
        data_sub.drop_duplicates(subset=['fraction_edited_eff'], inplace=True)

        #If multiple keys exist, take first value
        if len(data_sub) > 1:
            df.loc[ind, 'efficiency_full_grna_reported'] = data_sub.loc[:,
                                                                        'fraction_edited_eff'][0]
            df.loc[ind, 'total_count_reported_efficiency'] = data_sub.loc[:,
                                                                        'total_count'][0]
            df.loc[ind, 'edited_count_reported_efficiency'] = data_sub.loc[:,
                                                                        'edited_count_eff'][0]
        else:
            df.loc[ind, 'efficiency_full_grna_reported'] = data_sub.loc[:,
                                                                        'fraction_edited_eff'].values
            df.loc[ind, 'total_count_reported_efficiency'] = data_sub.loc[:,
                                                                        'total_count'].values
            df.loc[ind, 'edited_count_reported_efficiency'] = data_sub.loc[:,
                                                                        'edited_count_eff'].values
 
    # Add repeating data to dfs
    df_pos_edits.loc[:, ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated_calculated', 'efficiency_data_total_count', 'efficiency_data_edited_count', 'efficiency_data_efficiency_full_grna', 'efficiency_full_grna_reported', 'total_count_reported_efficiency', 'edited_count_reported_efficiency','total_count_reported_efficiency', 'edited_count_reported_efficiency']] = df.loc[:, [
        'original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated', 'efficiency_data_total_count', 'efficiency_data_edited_count', 'efficiency_data_efficiency_full_grna', 'efficiency_full_grna_reported', 'total_count_reported_efficiency', 'edited_count_reported_efficiency','total_count_reported_efficiency', 'edited_count_reported_efficiency']]

    #Save data
    os.chdir(saving_path + file_path_ext)
    df.to_csv('bystander_outcome_per_position.csv')
    df_pos_edits.to_csv('bystander_edit_per_position.csv')

#######################################################
# Pallaseni
#######################################################
# Parse pallaseni study data and process it
def parse_pallaseni_data(file_name, abe_efficiency_file_name, cbe_efficiency_file_name, num_positions_before_protospacer, num_positions, columns, file_path, eff_data_path, oligo_data_path, saving_path, file_path_ext):
    """
    Parse pallaseni study data and process it
    @params:
        file_name                            - Required  : file name of the data file (str)
        abe_efficiency_file_name             - Required  : file name of the abe efficiency data file (str)
        cbe_efficiency_file_name             - Required  : file name of the cbe efficiency data file (str)
        num_positions_before_protospacer     - Required  : number of unique base positions before protospacer index (int)
        num_positions                        - Required  : number of unique base positions after protospacer index (int)
        columns                              - Required  : column names for the outcome data frame (list)
        file_path                            - Required  : file path to the raw data file (str)
        eff_data_path                        - Required  : file path to the raw effiency data (str)
        oligo_data_path                      - Required  : file path to the oligo data (str)
        saving_path                          - Required  : file path to save the processed data file (str)
        file_path_ext                        - Required  : file path for folder route (str)
    """
    # Read data
    os.chdir(file_path + file_path_ext)
    data_by = pd.read_csv(file_name, index_col=0)

    # Read oligo txt file - starting G is 1 and not included in sequence only in gRNA
    oligo_data = pd.read_csv(oligo_data_path, index_col=0)

    # Read efficiency data that was shared via email
    os.chdir(eff_data_path)
    CBE_efficiency = pd.read_csv(cbe_efficiency_file_name, sep='\t', index_col=0)
    ABE_efficiency = pd.read_csv(abe_efficiency_file_name, sep='\t', index_col=0)

    # Generate position column names for position outcome edits
    pos_columns = [f"Position_{i}" for i in range(-num_positions_before_protospacer, num_positions+1)]
    pos_columns = [[f"{i} A", f"{i} T", f"{i} C", f"{i} G"] for i in pos_columns]
    pos_columns = [x for xs in pos_columns for x in xs]
    columns = columns + pos_columns
    
    #Create empty data frame
    df = pd.DataFrame(columns=columns)

    #Fixed editors according to study
    K562_editors = ['BE4-1', 'FNLS', 'ABERA']

    # Loop through data_by index and obtain oligo id
    pos_cols = [c for c in data_by.columns if c.startswith('W')]
    for i, ind in enumerate(data_by.index):
        print(f"Unpacking {i} / {len(data_by)}")
        #Unwrap data
        oligo = data_by.loc[ind, 'Oligo Id']
        total_count = data_by.loc[ind, 'Total Reads']
        base_editor = data_by.loc[ind, 'Editor']

        # Get oligo data
        grna = oligo_data.loc[oligo, 'Guide Sequence']
        target = oligo_data.loc[oligo, 'TargetSequence']
        pam_index = oligo_data.loc[oligo, 'PAM Index']
        sequence = target[pam_index - 30:][:40]
        protospace_position = pam_index - 20

        # Add editing frequency
        temp_data = data_by.loc[ind, pos_cols]
        for j, l in enumerate(list(sequence)):
            temp = temp_data.iloc[j*12:(j+1)*12]
            temp = temp.loc[(temp.values != 0)]

            for temp_ind in temp.index:
                nt_to = temp_ind[-1]
                freq = temp.loc[temp_ind]
                df.loc[i, f'Position_{j+1} {nt_to}'] = freq

        # Get right cell
        if base_editor in K562_editors:
            cell = 'K562'
        else:
            cell = 'HEK293T'

        # Include efficiency data
        try:
            if base_editor.startswith('AB'):
                efficiency = ABE_efficiency.loc[oligo, base_editor]
            else:
                efficiency = CBE_efficiency.loc[oligo, base_editor]
        except:
            efficiency = pd.NA

        #Assign values
        save_ind = i
        df.loc[save_ind, 'original_id'] = ind
        df.loc[save_ind, 'grna'] = grna
        df.loc[save_ind, 'sequence'] = sequence
        df.loc[save_ind, 'full_context_sequence'] = target
        df.loc[save_ind, 'protospace_position'] = protospace_position
        df.loc[save_ind, 'pam_index'] = pam_index
        df.loc[save_ind, 'efficiency_full_grna_reported'] = efficiency
        df.loc[save_ind, 'cell'] = cell
        df.loc[save_ind, 'base_editor'] = base_editor
        df.loc[save_ind, 'total_count_reported_efficiency'] = total_count
        df.loc[save_ind, 'total_count_reported_calculated'] = total_count

    #Save data
    os.chdir(saving_path + file_path_ext)
    df.to_csv('bystander_outcome_per_position.csv')

#######################################################
# Yuan
#######################################################
# Parse yuan study data and process it
def parse_yuan_data(file_name, num_positions, columns, file_path, oligo_data_path, saving_path, file_path_ext):
    """
    Parse yuan study data and process it
    @params:
        file_name                            - Required  : file name of the data file (str)
        num_positions                        - Required  : number of unique base positions after protospacer index (int)
        columns                              - Required  : column names for the outcome data frame (list)
        file_path                            - Required  : file path to the raw data file (str)
        oligo_data_path                      - Required  : file path to the oligo data (str)
        saving_path                          - Required  : file path to save the processed data file (str)
        file_path_ext                        - Required  : file path for folder route (str)
    """
    #Change working directory
    os.chdir(file_path + file_path_ext)

    #Read data
    data_by = pd.read_csv(file_name, sep="\t")
    data_by.columns = ['Nucleotide', 'Group', 'sgRNA', 'Position', 'Efficinecy']

    # Read oligo txt file - starting G is 1 and not included in sequence only in gRNA
    oligo_data = pd.read_csv(oligo_data_path, index_col=0)

    # Generate position column names for position outcome edits
    pos_columns = [f"Position_{i}" for i in range(1, num_positions+1)]
    pos_columns = [[f"{i} A", f"{i} T", f"{i} C", f"{i} G"] for i in pos_columns]
    pos_columns = [x for xs in pos_columns for x in xs]
    columns = columns + pos_columns
    
    #Create empty data frame
    df = pd.DataFrame(columns=columns)

    # Loop through data_by index and obtain oligo id
    for i, ind in enumerate(data_by.index):
        print(f"Unpacking {i} / {len(data_by)}")
        
        #Get oligo and base editor
        oligo = data_by.loc[ind, 'sgRNA']
        base_editor = data_by.loc[ind, 'Group']

        #Unwrap data by oligo 
        target = oligo_data.loc[oligo, 'TargetSequence']
        grna = oligo_data.loc[oligo, 'Guide Sequence']
        pam_index = oligo_data.loc[oligo, 'PAM Index']
        sequence = target[pam_index - 20:][:20]
        protospace_position = pam_index - 20
        editing_pos = data_by.loc[ind, 'Position']
        nt_to = data_by.loc[ind, 'Nucleotide']
        freq = data_by.loc[ind, 'Efficinecy']
        
        #Unique index
        save_ind = oligo + '_' + base_editor

        # Get right cell
        cell = 'HEK293T'

        #Assign values
        df.loc[save_ind, f'Position_{editing_pos} {nt_to}'] = freq
        df.loc[save_ind, 'original_id'] = save_ind
        df.loc[save_ind, 'grna'] = grna
        df.loc[save_ind, 'sequence'] = sequence
        df.loc[save_ind, 'full_context_sequence'] = target
        df.loc[save_ind, 'protospace_position'] = protospace_position
        df.loc[save_ind, 'pam_index'] = pam_index
        df.loc[save_ind, 'cell'] = cell
        df.loc[save_ind, 'base_editor'] = base_editor

    #Save data
    os.chdir(saving_path + file_path_ext)
    df.to_csv('bystander_outcome_per_position.csv')

#######################################################
# Song
#######################################################
# Parse song study data and process it
def parse_song_data(file_name, editing_window_start, editing_window_end, num_positions_before_protospacer, num_positions, columns, file_path, saving_path, file_path_ext, rebase_outcome_proportions = True):
    """
    Parse song study data and process it
    @params:
        file_name                            - Required  : file name of the data file (str)
        editing_window_start                 - Required  : start base position of the editing window (int)
        editing_window_end                   - Required  : end base position of the editing window (int)
        num_positions_before_protospacer     - Required  : number of unique base positions before protospacer index (int)
        num_positions                        - Required  : number of unique base positions after protospacer index (int)
        columns                              - Required  : column names for the outcome data frame (list)
        file_path                            - Required  : file path to the raw data file (str)
        saving_path                          - Required  : file path to save the processed data file (str)
        file_path_ext                        - Required  : file path for folder route (str)
        rebase_outcome_proportions           - Required  : rebase outcome proportions to 1 (bool)
    """
    #Change working directory
    os.chdir(file_path + file_path_ext)

    #Read data
    data_by = pd.read_excel(file_name, sheet_name='Merged')
    
    #Assign unique index
    data_by.index = data_by.loc[:, "target sequence (total 30 bps = 4 bp neighboring sequence + 20 bp protospacer + 3 bp NGG PAM+ 3 bp neighboring sequence)"] + \
        '_' + \
        data_by.loc[:,'Total Read count \n (experiment1 + experiment2)'].astype(str)

    # Generate position column names for position outcome edits
    pos_columns = [f"Position_{i}" for i in range(-num_positions_before_protospacer, num_positions+1)]
    pos_columns = [[f"{i} A", f"{i} T", f"{i} C", f"{i} G"] for i in pos_columns]
    pos_columns = [x for xs in pos_columns for x in xs]
    columns_df = columns + pos_columns

    #Create empty data frame
    df = pd.DataFrame(columns=columns_df)

    #Generate position column names for position edits
    pos_columns = [f"Position_{i}" for i in range(-num_positions_before_protospacer, num_positions+1)]
    columns_df = columns + pos_columns

    #Create empty data frame
    df_pos_edits = pd.DataFrame(columns=columns_df)

    # Generate position indices helper
    pos_indexes = range(-num_positions_before_protospacer, num_positions+1)
    zero_pos = np.where(np.array(list(pos_indexes)) == 0)[0][0]
    edit_window_pos_start = zero_pos + editing_window_start
    edit_window_pos_end = zero_pos + editing_window_end

    #Loop through all data points
    protospace_position = 4 #fixed
    for i, ind in enumerate(data_by.index.unique()):
        print(f"Unpacking {i} / {len(data_by.index.unique())}")

        #Unwrap data
        temp_data = data_by.loc[[ind], :]
        target = temp_data.loc[:,"target sequence (total 30 bps = 4 bp neighboring sequence + 20 bp protospacer + 3 bp NGG PAM+ 3 bp neighboring sequence)"][0]
        pam_index = protospace_position + 20
        grna = target[protospace_position:][:20]
        total_count = temp_data.loc[:,'Total Read count \n (experiment1 + experiment2)'][0]
        edited_count = total_count - temp_data.loc[(temp_data.loc[:, 'Outcome seqeunce']== target), 'Outcome seqeunce read count\n(experiment1 + experiment2)'][0]
        scaling_factor_edited_count_denominator = total_count / edited_count
        or_seq = target
        outcomes = [list(out) for out in temp_data.loc[:, 'Outcome seqeunce']]
        outcomes = pd.DataFrame(outcomes)
        fraction_edited = temp_data.loc[:,'Outcome seqeunce read count\n(experiment1 + experiment2)'] / total_count

        #Rebase to 1 if set to true
        if rebase_outcome_proportions:
            fraction_edited = fraction_edited/fraction_edited.sum()

        efficiency = fraction_edited[temp_data.loc[:,'Outcome seqeunce'] != target].sum()
        #Loop through all outcomes
        for k, nt in enumerate(list(or_seq)):
            pos_index = pos_indexes[k]
            all_nt_pos = outcomes.iloc[:, k].unique()
            for nt_pos in all_nt_pos:
                # * scaling_factor
                df.loc[ind, f"Position_{pos_index} {nt_pos}"] = fraction_edited[outcomes.iloc[:, k].values == nt_pos].sum()

            # Add view editing percentage at position
            df_pos_edits.loc[ind,f"Position_{pos_index}"] = fraction_edited[outcomes.iloc[:, k].values != nt].sum()

        editing_window_efficiency = fraction_edited[np.any(outcomes.iloc[:, edit_window_pos_start:edit_window_pos_end].values != list(or_seq[edit_window_pos_start:edit_window_pos_end]), axis=1)].sum()

        grna = grna
        sequence = target
        genomic_context = "NA"  
        genomic_context_build = "NA"  
        cell = "HEK293T" 
        base_editor = temp_data.loc[:, 'Base Editor'][0]
        editing_windows_3_10_efficiency_reported = (data_by.loc[[ind],'Efficiency (%)'] / 100).values[0]

        #Assign data
        df.loc[ind, ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'scaling_factor_edited_count_denominator',
                    f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated', 'editing_windows_3_10_efficiency_reported']] = [ind, grna, or_seq, sequence, protospace_position, pam_index, genomic_context, genomic_context_build, cell, base_editor, total_count, edited_count, efficiency, scaling_factor_edited_count_denominator, editing_window_efficiency, editing_windows_3_10_efficiency_reported]
    
    # Add repeating data to dfs
    df.loc[:, 'total_count_reported_efficiency'] = df.loc[:, 'total_count']
    df.loc[:, 'edited_count_reported_efficiency'] = df.loc[:, 'edited_count']
    df_pos_edits.loc[:, ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated','editing_windows_3_10_efficiency_reported','total_count_reported_efficiency','edited_count_reported_efficiency']] = df.loc[:, [
        'original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated','editing_windows_3_10_efficiency_reported', 'total_count_reported_efficiency','edited_count_reported_efficiency']]

    #Save data
    os.chdir(saving_path + file_path_ext)
    df.to_csv(f'bystander_outcome_per_position_rebased_{rebase_outcome_proportions}.csv')
    df_pos_edits.to_csv(f'bystander_edit_per_position_rebased_{rebase_outcome_proportions}.csv')
