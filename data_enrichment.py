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
from Bio.SeqUtils import MeltingTemp
import itertools
import requests
from Bio.Align import PairwiseAligner
from Bio import SeqIO
import subprocess
from subprocess import CREATE_NEW_CONSOLE
import pandas as pd
import os
import numpy as np

#####################################################################################################################################################################
# Data Enrichment
#####################################################################################################################################################################
#######################################################
# cas-offinder (requires installation of cas-offinder)
# source: https://github.com/snugel/cas-offinder
#######################################################
#Run cas-offinder to find matching locations on assembly
def run_cas_offinder(authors, max_mismatches, max_mismatches_ratio, assembly_path, processed_data_path, savings_path, cas_offinder_path, task):
    """
    Run cas-offinder to find matching locations on assembly
    @params:
        authors                              - Required  : authors/studies to run (list)
        max_mismatches                       - Required  : maximum number of mismatches (int)
        max_mismatches_ratio                 - Required  : maximum percentage of mismatches - ignored if set to 0 (float)
        assembly_path                        - Required  : assembly storage path relative to cas-offinder installation directory (str)
        processed_data_path                  - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save the cas-offinder files (str)
        cas_offinder_path                    - Required  : file path to the cas-offinder executable (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
    """
    #Loop through authors/studies
    for author in authors:
        #Empty list for sequences
        checked_seq = []

        # Set working directoy
        os.chdir(processed_data_path + '\\' + author + '\\' + task)

        # Check if vF with chromosome location already exists --> different routine
        vf_file_names = [f for f in os.listdir() if f.endswith('vF.csv') or f.endswith('vLocation.csv')]

        # File name of cas-offinder output file 
        output_file_name = f"out_{author}_sequence.txt"

        # Determine unique sequences - if location data already exist consider only non-matched sequences
        if len(vf_file_names) > 0:
            bystander_file_name = vf_file_names[0]
            data = pd.read_csv(bystander_file_name, index_col=0)
            seq_data = data.loc[(pd.isnull(data.mismatches) == True), :].sequence.unique()
        else:
            bystander_file_name = [f for f in os.listdir() if f.startswith('bystander')][0]
            data = pd.read_csv(bystander_file_name, index_col=0)
            seq_data = data.sequence.unique()

            # Check if existing output exists already
            os.chdir(savings_path)
            exis_files = [f for f in os.listdir() if f.startswith(output_file_name[:10])]

            #Check sequences in existing files
            for f in exis_files:
                print(f"Loading existing output file {f}")

                # Batch reading csv
                start_row = 1
                counter_reads = 0
                rows_per_read = 100000
                while True:
                    exis_data = pd.read_csv(f, sep="\t", skiprows=start_row, nrows=rows_per_read)

                    checked_seq = checked_seq + list(exis_data.crRNA.unique())

                    # Adjust start row
                    counter_reads += 1
                    start_row = counter_reads * rows_per_read

                    print(f"Lenght checked sequence {len(checked_seq)} - load {counter_reads}")

                    if len(exis_data) < rows_per_read:
                        break

        if len(seq_data) == 0:
            print(f"Skipping {author}")
            continue

        #Construct input data txt file
        input_data = f"{assembly_path}\n" 
        seq_counter = 0
        for i, seq in enumerate(seq_data):
            if i == 0:
                input_data += "N"*len(seq) + '\n'

                if max_mismatches_ratio != 0:
                    max_mismatches = int(max_mismatches_ratio*len(seq))

            if seq not in checked_seq:
                seq_counter += 1
                input_data += f"{seq} {max_mismatches}\n"

        print(f"Author {author}: sequences to check {(seq_counter)} / {len(seq_data)}")

        # Save cas-offinder input file
        os.chdir(cas_offinder_path + '\\' + 'input')
        input_file = open(f"input_{author}_sequence.txt", "w")
        n = input_file.write(input_data)
        input_file.close()
        
        #Terminal command for cas-offinder
        cas_cmd = f"./cas-offinder.exe ./input/input_{author}_sequence.txt G E:/cas-offinder_output/out_{author}_sequence_{max_mismatches}.txt"

        #Run cas-offinder in terminal
        subprocess.Popen(f'powershell.exe cd \'{cas_offinder_path}\' \n{cas_cmd}', creationflags=CREATE_NEW_CONSOLE)

#Add cas-offinder results to data files
def add_cas_offinder_results(authors, location_cols, processed_data_path, savings_path, task):
    """
    Add cas-offinder results to data files
    @params:
        authors                              - Required  : authors/studies to run (list)
        location_cols                        - Required  : new location columns to be added (list)
        processed_data_path                  - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save the cas-offinder files (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
    """
    #Loop through all authors/studies
    for author in authors:
        # Load bystander file
        os.chdir(processed_data_path + '\\' + author + '\\' + task)
        bystander_file_names = [f for f in os.listdir() if f.startswith('bystander') and f.endswith('vLocation.csv')]

        if len(bystander_file_names) == 0:
            bystander_file_names = [f for f in os.listdir() if f.startswith('bystander')]

        bystander_file_name = bystander_file_names[0]
        data = pd.read_csv(bystander_file_name, index_col=0)
        data.index = data.sequence

        # Initialize columns
        if 'chromosome' not in data.columns:
            data[location_cols] = pd.NA

        # Get data frame with start and end location for arbaba to narrow down matches with given information
        if author == 'Arbab':
            hg19_ranges = data[(data.genomic_context_build =='hg19')].genomic_context
            hg19_vals = []
            for v in hg19_ranges.values:
                try:
                    hg19_vals.append(int(str(v).split(':')[1].split('_')[0]))
                except:
                    hg19_vals.append(pd.NA)

            hg19_ranges = pd.DataFrame(hg19_vals, index=hg19_ranges.index, columns=['start'])
            hg19_ranges['end'] = hg19_ranges['start'] + len(hg19_ranges.index[0])

            hg38_ranges = data[(data.genomic_context_build =='hg38')].genomic_context
            hg38_start_vals = []
            hg38_end_vals = []
            for v in hg38_ranges.values:
                try:
                    hg38_start_vals.append(int(str(v).split(':')[1].split('-')[0]))
                    hg38_end_vals.append(int(str(v).split(':')[1].split('-')[1]))
                except:
                    hg38_start_vals.append(pd.NA)
                    hg38_end_vals.append(pd.NA)

            hg38_ranges = pd.DataFrame(hg38_start_vals, index=hg38_ranges.index, columns=['start'])
            hg38_ranges['end'] = hg38_end_vals

            hg19_ranges.dropna(inplace=True)
            hg38_ranges.dropna(inplace=True)

        output_file_name = f"out_{author}_sequence.txt"

        # Check if existing output exists already
        os.chdir(savings_path)
        exis_files = [f for f in os.listdir() if f.startswith(output_file_name[:10])]

        #Loop through sequence matching files
        for f in exis_files:
            print(f"Loading existing output file {f}")

            # Arbab specific routine
            if author == 'Arbab':
                if 'hg19' in f:
                    assembly_type = 'hg19'
                else:
                    assembly_type = 'hg38'
            else:
                assembly_type = 'hg38'

            # Batch reading csv
            start_row = 1
            counter_reads = 0
            rows_per_read = 20000000
            while True:
                print(f"Loading existing output file {f} - row {start_row} - author {author} - unfilled {sum(pd.isnull(data.assembly_sequence))}")
                if start_row == 1:
                    exis_data = pd.read_csv(f, sep="\t", skiprows=start_row, nrows=rows_per_read)
                    temp_cols = exis_data.columns
                else:
                    exis_data = pd.read_csv(f, sep="\t", skiprows=start_row, nrows=rows_per_read, header=None)
                    exis_data.columns = temp_cols

                exis_data.index = exis_data.crRNA

                #Loop through matches
                for ind in exis_data.index.unique():
                    loc_data = exis_data.loc[[ind], :]
                    
                    # Check if location in chromosom range specified by author - narrow down matches with given data
                    if author == 'Arbab':
                        #Arbab routine to check assembly type as hg19 and hg38 is used
                        if assembly_type == 'hg19':
                            if ind in hg19_ranges.index:
                                # Sequence are unique so max 1 match
                                temp_hg19_ranges = hg19_ranges.loc[[ind], :]
                                start_seq = temp_hg19_ranges.start[0]
                                end_seq = temp_hg19_ranges.end[0]
                                loc_data = loc_data.loc[(loc_data.Location >= start_seq) & (
                                    loc_data.Location <= end_seq), :]

                                if len(loc_data) == 0:
                                    continue
                            else:
                                continue
                        elif assembly_type == 'hg38':
                            if ind in hg38_ranges.index:
                                # Sequence are unique so max 1 match
                                temp_hg38_ranges = hg38_ranges.loc[[ind], :]
                                start_seq = temp_hg38_ranges.start[0]
                                end_seq = temp_hg38_ranges.end[0]
                                loc_data = loc_data.loc[(loc_data.Location >= start_seq) & (
                                    loc_data.Location <= end_seq), :]

                                if len(loc_data) == 0:
                                    continue
                            else:
                                continue

                        loc_data.sort_values(["Mismatches"], ascending=True, inplace=True)
                        chrom = loc_data.Chromosome[0]
                        location = loc_data.Location[0]
                        direction = loc_data.Direction[0]
                        mismatches = loc_data.Mismatches[0]
                        assembly_sequence = loc_data.DNA[0]
                        not_unique = sum(loc_data.Mismatches == mismatches) > 1

                        #Check if match is already added and compare if match is better
                        if not pd.isnull(data[(data.genomic_context_build == assembly_type)].loc[[ind], 'chromosome'][0]):
                            sav_mismatches = data[(data.genomic_context_build == assembly_type)].loc[[ind], 'mismatches'][0]
                            sav_location = data[(data.genomic_context_build == assembly_type)].loc[[ind], 'location'][0]
                            if (sav_mismatches < mismatches) or ((sav_mismatches == mismatches) and (sav_location != location)):
                                data[(data.genomic_context_build == assembly_type)].loc[[ind], 'not_unique_match_flag'] = True

                                # Skip sequence from adding
                                continue

                        #Assign values
                        data.loc[(data.genomic_context_build == assembly_type) & (
                            data.index == ind), 'assembly'] = assembly_type
                        data.loc[(data.genomic_context_build == assembly_type) & (
                            data.index == ind), 'chromosome'] = chrom
                        data.loc[(data.genomic_context_build == assembly_type) & (
                            data.index == ind), 'location'] = location
                        data.loc[(data.genomic_context_build == assembly_type) & (
                            data.index == ind), 'direction'] = direction
                        data.loc[(data.genomic_context_build == assembly_type) & (
                            data.index == ind), 'mismatches'] = mismatches
                        data.loc[(data.genomic_context_build == assembly_type) & (
                            data.index == ind), 'assembly_sequence'] = assembly_sequence
                        data.loc[(data.genomic_context_build == assembly_type) & (
                            data.index == ind), 'not_unique_match_flag'] = not_unique

                    else:
                        #Sort best matches
                        loc_data.sort_values(
                            ["Mismatches"], ascending=True, inplace=True)

                        #Get values
                        chrom = loc_data.Chromosome[0]
                        location = loc_data.Location[0]
                        direction = loc_data.Direction[0]
                        mismatches = loc_data.Mismatches[0]
                        assembly_sequence = loc_data.DNA[0]
                        not_unique = sum(loc_data.Mismatches == mismatches) > 1

                        #Check if match is already added and compare if match is better
                        if not pd.isnull(data.loc[[ind], 'chromosome'][0]):
                            sav_mismatches = data.loc[[ind], 'mismatches'][0]
                            sav_location = data.loc[[ind], 'location'][0]
                            if (sav_mismatches < mismatches) or ((sav_mismatches == mismatches) and (sav_location != location)):
                                data.loc[[ind], 'not_unique_match_flag'] = True

                                # Skip sequence from adding
                                continue

                        #Assign values
                        data.loc[ind, 'assembly'] = assembly_type
                        data.loc[ind, 'chromosome'] = chrom
                        data.loc[ind, 'location'] = location
                        data.loc[ind, 'direction'] = direction
                        data.loc[ind, 'mismatches'] = mismatches
                        data.loc[ind, 'assembly_sequence'] = assembly_sequence
                        data.loc[ind, 'not_unique_match_flag'] = not_unique

                # Adjust start row
                counter_reads += 1
                start_row = start_row + rows_per_read

                if len(exis_data) < rows_per_read:
                    break

        #Arbab check
        if author != 'Arbab':
            data.loc[:, 'assembly'] = 'hg38'

        # Save data file
        os.chdir(savings_path)
        data.reset_index(drop=True, inplace=True)
        if 'vLocation' in bystander_file_name.split('.')[0]:
            data.to_csv(bystander_file_name.split('.')[0]+'.csv')
        else:
            data.to_csv(bystander_file_name.split('.')[0]+'_vLocation'+'.csv')

        # Add data to second data file
        if len(bystander_file_names) > 1:
            data_2 = pd.read_csv(bystander_file_names[1], index_col=0)
            data_2.index = data.sequence
            data_2.loc[data.index, location_cols] = data.loc[:, location_cols]
            data_2.to_csv(bystander_file_names[1].split('.')[0]+'_vLocation'+'.csv')

#######################################################
# Biopython
#######################################################
#Match sequence location via biopython - only used for arbab as chromosome and location is approximately known
def match_location_via_biopython(author, assembly_path, processed_data_path, savings_path, task):
    """
    Match sequence location via biopython - only used for arbab as chromosome and location is approximately known
    @params:
        author                               - Required  : author/study to run (str)
        assembly_path                        - Required  : assembly storage path relative to cas-offinder installation directory (str)
        processed_data_path                  - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save the cas-offinder files (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
    """
    # Set working directoy
    os.chdir(processed_data_path + '\\' + author + '\\' + task)

    # Read data
    file_name = [f for f in os.listdir() if f.startswith('bystander') and f.endswith('.csv')][0]
    data = pd.read_csv(file_name, index_col=0)

    # Initialize columns
    if 'chromosome' not in data.columns:
        data['assembly'] = pd.NA
        data['chromosome'] = pd.NA
        data['location'] = pd.NA
        data['direction'] = pd.NA
        data['mismatches'] = pd.NA
        data['assembly_sequence'] = pd.NA
        data['not_unique_match_flag'] = pd.NA

    #Create chromosome dataframe with start and end values to loop through
    chromosomes = list(set([str(c).split(':')[0] for c in data.genomic_context]))
    chromosomes.remove('nan')
    assemblies = data.genomic_context_build.unique()

    chrom_ranges = data.genomic_context
    chrom_ranges = data.loc[pd.isnull(data.assembly_sequence), 'genomic_context']
    ind_chrom_ranges = data.loc[pd.isnull(data.assembly_sequence), 'genomic_context'].index
    chrom_vals = []
    for v in chrom_ranges.values:
        try:
            chrom_vals.append([str(v).split(':')[0], int(
                str(v).split(':')[1].split('_')[0])])
        except:
            try:
                chrom_vals.append([str(v).split(':')[0], int(
                    str(v).split(':')[1].split('-')[0])])
            except:
                chrom_vals.append([pd.NA, pd.NA])

    chrom_ranges = pd.DataFrame(chrom_vals, index=ind_chrom_ranges, columns=['chrom', 'start'])
    chrom_ranges['end'] = chrom_ranges['start'] + 100
    chrom_ranges['assembly'] = data.loc[chrom_ranges.index,
                                        'genomic_context_build']
    chrom_ranges.dropna(subset=['chrom'], inplace=True)

    #Loop througn assemblies and chromosome
    for assembly in assemblies:
        for chr_name in chromosomes:
            print(f'Chr {chr_name} - assembly {assembly}')
            ref_genome_path = fr"{assembly_path}\{assembly}.chromFa\chroms\{chr_name}.fa"
            
            #Load fasta file
            chrom = SeqIO.parse(ref_genome_path, 'fasta')
            chrom = str(next(chrom).seq)

            #Get sequences to check that fall within search window
            ind_to_check = chrom_ranges.loc[(chrom_ranges.chrom == chr_name) & (chrom_ranges.assembly == assembly), :].index
            for ind in ind_to_check:
                #Unwrap data
                pos = chrom_ranges.loc[ind, 'start'] - 75  # enlarge vie
                sequence = data.loc[ind, 'sequence']

                #Get forward and reverse sequence
                temp_chrom = chrom[pos:pos+250]
                temp_chrom_forward = temp_chrom.upper()
                temp_chrom_reverse = temp_chrom_forward[::-1]

                # Align sequences
                try:
                    aligner = PairwiseAligner()
                    aligner.mode = 'local'
                    aligner.mismatch_score = -1
                    aligner.gap_score = -10
                    alignments_fwd = aligner.align(temp_chrom_forward, sequence)
                    alignments_rvs = aligner.align(temp_chrom_reverse, sequence)

                    alignment_fwd = alignments_fwd[0]
                    alignment_fwd.score
                    alignment_rvs = alignments_rvs[0]
                    alignment_rvs.score

                    #Compare forward and reverse score and take better fit
                    if alignment_fwd.score >= alignment_rvs.score:
                        direction = '+'

                        align_graph = str(alignment_fwd).split('\n')
                        chrom_seq, indication, target = align_graph[0], align_graph[1], align_graph[2]

                        location_start = target.count(' ')
                        assembly_sequence = list(temp_chrom_forward[location_start:location_start+len(sequence)])

                    else:
                        direction = '-'

                        align_graph = str(alignment_rvs).split('\n')
                        chrom_seq, indication, target = align_graph[0], align_graph[1], align_graph[2]

                        location_start = target.count(' ')
                        assembly_sequence = list(temp_chrom_reverse[location_start:location_start+len(sequence)])

                    # Match is left outerbound -> skip
                    if chrom_seq.find(' ') != -1:
                        print('Skipping outerbound')
                        continue

                    # Calculate mismatches
                    mismatches = 0
                    for i, c in enumerate(assembly_sequence):
                        if c != sequence[i]:
                            mismatches += 1
                            assembly_sequence[i] = assembly_sequence[i].lower()

                    assembly_sequence = "".join(assembly_sequence)
                    location = int(pos + location_start)

                    # Assign data
                    data.loc[ind, 'assembly'] = assembly
                    data.loc[ind, 'chromosome'] = chr_name
                    data.loc[ind, 'location'] = location
                    data.loc[ind, 'direction'] = direction
                    data.loc[ind, 'mismatches'] = mismatches
                    data.loc[ind, 'assembly_sequence'] = assembly_sequence
                    data.loc[ind, 'not_unique_match_flag'] = False

                except Exception as e:
                    print(f"{ind} - {e}")

    #Save data
    os.chdir(savings_path)
    data.to_csv(file_name.split('.csv')[0]+'_vLocation_Biopython.csv')

#######################################################
# SCREEN
# source: https://screen-beta-api.wenglab.org
#######################################################
#Get encode screen data
def get_encode_data(chr, assembly='GRCh38', step_size=1e6, max_bp_length=500e6):
    """
    Get encode screen data
    @params:
        chr                        - Required  : chromosome - example: chr1 (str)
        assembly                   - Required  : assembly to use - example: GRCh38 (str)
        step_size                  - Required  : step size for downloading data (int)
        max_bp_length              - Required  : maximum base length to trigger download stop (int)
    """
    # Request headers
    headers = {
        "accept": "application/json",
        "accept-language": "en-GB,en;q=0.9,en-US;q=0.8,de;q=0.7",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "pragma": "no-cache",
        "sec-ch-ua": "\"Chromium\";v=\"104\", \" Not A;Brand\";v=\"99\", \"Google Chrome\";v=\"104\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
        "Referer": "https://screen.encodeproject.org/",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }
    url = "https://screen-beta-api.wenglab.org/dataws/cre_table"

    # Empty df
    data = pd.DataFrame()

    # Request arguments
    start_coord = 0
    end_coord = int(step_size)

    # Loop through all runs
    counter = 0
    while True:
        print(f'Downloading {chr} - {start_coord}:{end_coord} - results {len(data)}')

        #Build request body
        body = f"{{\"uuid\":\"{np.random.randint(1,100000)}\",\"assembly\":\"{assembly}\",\"accessions\":[],\"coord_chrom\":\"{chr}\",\"coord_start\":{start_coord},\"coord_end\":{end_coord},\"gene_all_start\":0,\"gene_all_end\":5000000,\"gene_pc_start\":0,\"gene_pc_end\":5000000,\"rank_dnase_start\":1.64,\"rank_dnase_end\":10,\"rank_promoter_start\":-10,\"rank_promoter_end\":10,\"rank_enhancer_start\":-10,\"rank_enhancer_end\":10,\"rank_ctcf_start\":-10,\"rank_ctcf_end\":10,\"cellType\":null,\"element_type\":null}}"
        
        req = requests.post(url=url, data=body, headers=headers)
        if req.status_code == 200:
            req = req.json()
        else:
            print(f"Error requesting data. Status code: {req.status_code} - {req.text}")
            break

        #Unpack data and append to master df
        temp_data = pd.json_normalize(req['cres'])
        data = pd.concat([data, temp_data], axis=0)

        # Adjust counter
        counter += 1
        start_coord = end_coord
        end_coord = int(end_coord + step_size)

        #Stop condition
        if start_coord >= max_bp_length:
            print("Max bp reached")
            break

    return data

#######################################################
# RNA fold (requires installation of RNAfold)
# source: https://www.tbi.univie.ac.at/RNA/
#######################################################
#Run RNAfold to calculate MFE energy term
def run_RNA_fold(authors, processed_data_path, savings_path, RNA_fold_path, task = "Bystander"):
    """
    Run RNAfold to calculate MFE energy term
    @params:
        authors                              - Required  : authors/studies to run (list)
        processed_data_path                  - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save the RNAfold files (str)
        RNA_fold_path                        - Required  : file path to the RNAfold executable (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
    @requirements:
        RNAfold installation                 - Source: https://www.tbi.univie.ac.at/RNA/
    """
    #Init input data
    input_data = ""

    #Loop through authors/studies
    for author in authors:
        # Set working directoy
        os.chdir(processed_data_path + '\\' + author + '\\' + task)

        #Read data
        file_name = [f for f in os.listdir() if f.startswith('bystander') and f.endswith('.csv')][0]
        data = pd.read_csv(file_name, index_col=0)
        seq_data = data.grna.unique()
        
        #Construct input data txt file
        seq_counter = 0
        for i, seq in enumerate(seq_data):
            seq_counter += 1
            input_data += f"{seq}\n"

        print(f"Author {author}: sequences to check {(seq_counter)} / {len(seq_data)}")

    print(f"Total gRNA length " + str(input_data.count('\n')))

    #Save RNAfold input file
    os.chdir(savings_path)
    input_file = open(f"input.txt", "w")
    n = input_file.write(input_data)
    input_file.close()

    #Terminal command for RNAfold
    cmd = "./RNAfold.exe -i " + str(savings_path).replace("\\", "/") + "/input.txt --outfile=output.txt"

    #Run RNAfold in terminal
    subprocess.Popen(f'powershell.exe cd \'{RNA_fold_path}\' \n{cmd}', creationflags=CREATE_NEW_CONSOLE)

#Unpack RNA fold output into a data frame
def unpack_RNA_fold_output(file_path, savings_path):
    """
    Unpack RNA fold output into a data frame
    @params:
        file_path                            - Required  : file path to the RNAfold output data file (str)
        savings_path                         - Required  : file path to save the RNAfold data frame (str)
    """
    #Load data
    data = pd.read_csv(file_path, sep='\n', header=None)

    #Create empty data frame
    df = pd.DataFrame(columns=['grna', 'free_energy'])

    #Init variables
    energy = None
    grna = None

    #Loop through all data indices
    for ind in data.index:
        output_val = data.loc[ind, :]

        #String handling to extract RNA values
        if "(" in str(output_val):
            energy = str(data.loc[ind, :].astype(str))
            energy = float(energy.split('(')[-1].split(')')[0].replace(' ', ''))
        else:
            grna = output_val
            energy = None

        if energy != None:
            #Assign values to df
            ind_df = len(df)
            df.loc[ind_df, 'grna'] = grna[0]
            df.loc[ind_df, 'free_energy'] = energy

            grna = None
            energy = None

    #Save data
    df.drop_duplicates(inplace=True)
    os.chdir(savings_path)
    df.to_csv('RNAfold_output.csv')

#######################################################
# CRISPRspec (requires CRISPRspec package)
# source: https://github.com/RTH-tools/crisproff
#######################################################
#Run CRISPRspec to calculate free energy terms
def run_CRISPR_spec(authors, param_values, rel_comb, processed_data_path, savings_path, crispr_spec_path, task):
    """
    Run CRISPRspec to calculate free energy terms
    @params:
        authors                              - Required  : authors/studies to run (list)
        param_values                         - Required  : potential CRISPRspec variable values used to calculate combinations (list(list))
        rel_comb                             - Required  : relevant parameter combinations to consider - subsetting all parameter combinations (list)
        processed_data_path                  - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save the CRISPRspec files (str)
        crispr_spec_path                     - Required  : file path to the CRISPRspec git (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
    @requirements:
        CRISPRspec installation              - Source: https://github.com/RTH-tools/crisproff
    """
    #Init CRISPRspec
    os.chdir(crispr_spec_path)
    from CRISPRspec_CRISPRoff_pipeline import read_energy_parameters, calcRNADNAenergy, get_eng
    read_energy_parameters(ENERGY_MODELS_PICKLE_FILE="energy_dics.pkl")

    #Create empty data frame
    seq_df = pd.DataFrame(columns=['grna', 'target'])

    # Loop through all author/studies
    for author in authors:
        #Set working directory
        os.chdir(processed_data_path + '\\' + author + '\\' + task)

        #Read data
        file_name = [f for f in os.listdir() if f.startswith('bystander') and f.endswith('.csv')][0]
        data = pd.read_csv(file_name, index_col=0)

        #Unwrap grna and protospacer sequences
        vals = []
        for ind in data.index:
            guide = data.grna[ind]
            seq = data.loc[ind, 'full_context_sequence']
            proto_index = data.loc[ind, 'protospace_position']

            vals.append([guide, seq[proto_index:][:20]])
        
        seq_df = pd.concat([seq_df, pd.DataFrame(vals, columns=['grna', 'target'])], axis=0)

    seq_df.drop_duplicates(inplace=True)
    seq_df.reset_index(drop=True, inplace=True)

    #Create empty data frame for free energy terms
    df = pd.DataFrame(columns=['grna', 'target'])

    #Generate relevant parameter combinations
    params = (list(itertools.product(*param_values)))
    params = [p for i, p in enumerate(params) if i in rel_comb]

    #Run crispr spec for all parameter combinations
    for ind in seq_df.index:
        print(f"{ind} / {len(seq_df)}")
        try:
            grna = seq_df.loc[ind, 'grna']
            target = seq_df.loc[ind, 'target']

            ind_df = len(df)

            #Calculate energy terms for all parameter combinations
            for i, param in enumerate(params):
                temp_energy = get_eng(grna, target, calcRNADNAenergy, GU_allowed=param[0], pos_weight=param[1], pam_corr=param[2], grna_folding=param[3], dna_opening=param[4], dna_pos_wgh=param[5])
                df.loc[ind_df, f'energy_{i+1}'] = temp_energy

            #Assign values
            df.loc[ind_df, 'grna'] = grna
            df.loc[ind_df, 'target'] = target
        except Exception as e:
            print(f"Error: {grna} - {target} {e}")

    #Save data
    os.chdir(savings_path)
    df.to_csv('crisproff_energies_all_combinations.csv')

#######################################################
# Melting Temperature
#######################################################
#Calculate melting temperatures
def calculate_melting_temperatures(authors, processed_data_path, savings_path, task):
    """
    Calculate melting temperatures
    @params:
        authors                              - Required  : authors/studies to run (list)
        processed_data_path                  - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save the melting temperature files (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
    """
    #Create empty data frame
    seq_df = pd.DataFrame(columns=['grna', 'target'])
    
    #Loop through all authors/studies
    for author in authors:
        #Set working directory
        os.chdir(processed_data_path + '\\' + author + '\\' + task)

        #Read data
        file_name = [f for f in os.listdir() if f.startswith('bystander') and f.endswith('.csv')][0]
        data = pd.read_csv(file_name, index_col=0)

        #Unwrap grna and protospacer sequences and create a data frame
        vals = []
        ind = 0
        for ind in data.index:
            guide = data.grna[ind]
            seq = data.loc[ind, 'full_context_sequence']
            proto_index = data.loc[ind, 'protospace_position']

            vals.append([guide, seq[proto_index:][:20]])
        seq_df = pd.concat([seq_df, pd.DataFrame(vals, columns=['grna', 'target'])], axis=0)
    
    seq_df.drop_duplicates(inplace=True)
    seq_df.reset_index(drop=True, inplace=True)

    #Create empty data frame
    df = pd.DataFrame(columns=['grna', 'target', 'melt_temperature_grna', 'melt_temperature_target'])

    #Calculate melting temperatures for all targets and grnas
    for ind in seq_df.index:
        print(f"{ind} / {len(seq_df)}")
        try:
            grna = seq_df.loc[ind, 'grna']
            target = seq_df.loc[ind, 'target']

            melt_temperature_grna = MeltingTemp.Tm_NN(grna)
            melt_temperature_target = MeltingTemp.Tm_NN(target)

            #Assign values
            ind_df = len(df)
            df.loc[ind_df, 'grna'] = grna
            df.loc[ind_df, 'target'] = target
            df.loc[ind_df, 'melt_temperature_grna'] = melt_temperature_grna
            df.loc[ind_df, 'melt_temperature_target'] = melt_temperature_target
        except Exception as e:
            print(f"Error: {grna} - {target} {e}")

    #Save data
    os.chdir(savings_path)
    df.to_csv('melting_temperature.csv')

#######################################################
# Add Data and Harmonize
#######################################################
# Harmonize data and add padding to the target sequence
def harmonize_data_padding(authors, remove_cols, data_path, grna_length=20, task='Bystander', latest_file_suffix="vLocation", suffix_savings="vF"):
    """
    Harmonize data and add padding to the target sequence
    @params:
        authors                              - Required  : authors/studies to run (list)
        remove_cols                          - Required  : columns to remove from data files (list)
        data_path                            - Required  : file path to the processed data files (str)
        grna_length                          - Required  : length of the used grna (int)
        task                                 - Required  : machine learning task - used in the folder structure (str)
        latest_file_suffix                   - Required  : suffix of file version to use (str)
        suffix_savings                       - Required  : suffix that should be used for saving the file (str)
    """
    #Init empty paddings dict
    paddings = {}

    #Determine max right and left pad
    for author in authors:
        #Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        file_names = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Load data
        f = file_names[0]
        data = pd.read_csv(f, index_col=0)

        #Calculate sequence specific variables
        left_pad = min_protospace_pos = data.protospace_position.min() 
        max_protospace_pos = data.protospace_position.max()
        max_length_sequence = max([len(s) for s in data.full_context_sequence])
        right_pad = max_length_sequence - (min_protospace_pos + 20)

        pos_numbers = list(set([int(c.split(' ')[0].split('_')[1]) for c in data.columns if c.startswith("Position")]))
        left_pad_pos = min_pos = abs(min(pos_numbers))
        max_pos = max(pos_numbers)
        right_pad_pos = max_pos - 20 + 1  # Starts at 0

        #Add paddings details for every study
        paddings[author] = {
            'min_protospace_pos': min_protospace_pos,
            'max_protospace_pos': max_protospace_pos,
            'max_length_sequence': max_length_sequence,
            'left_pad_seq': left_pad,
            'right_pad_seq': right_pad,
            'min_pos': min_pos,
            'max_pos': max_pos,
            'left_pad_pos': left_pad_pos,
            'right_pad_pos': right_pad_pos,
        }

    # Build new harmonized dataframe
    # Symmetric padding leftpad = rightpad
    seq_pads = []
    pos_pads = []
    for k, v in paddings.items():
        seq_pads.append(v['left_pad_seq'])
        seq_pads.append(v['right_pad_seq'])

        pos_pads.append(v['left_pad_pos'])
        pos_pads.append(v['right_pad_pos'])

    seq_padding = max(seq_pads)
    pos_padding = max(pos_pads)

    # Determine max right and left pad
    padded_seq = None
    padded_context = None
    placeholder_sign = 'N'

    #Paddings for sequences and Position ranges
    pos_columns = [f"Position_{i}" for i in range(-pos_padding, 19+pos_padding+1)]
    outcome_cols = [[f"{i} A", f"{i} T", f"{i} C", f"{i} G"] for i in pos_columns]
    outcome_cols = [x for xs in outcome_cols for x in xs]

    #Add paddings to all studies
    for author in authors:
        #Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        if padded_seq != None:
            padded_seq = None

        if padded_context != None:
            padded_context = None

        #Get all file names
        file_names = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Add paddings to all files
        for f in file_names:
            data = pd.read_csv(f, index_col=0)
            data.index = data.original_id

            seq_lengths = [len(s) for s in data.full_context_sequence]
            left_pad = [num*placeholder_sign for num in (seq_padding - data.protospace_position)]
            right_pad = [num*placeholder_sign for num in (1+(seq_padding) - (seq_lengths - (data.protospace_position + grna_length)))]

            # Handle bystander_outcome file
            if f.startswith('bystander_outcome'):
                cols = [c for c in data.columns if not c.startswith('Position')]
                new_cols = cols + outcome_cols
                temp_data = pd.DataFrame(columns=new_cols, index=data.index)
                temp_data.loc[data.index,data.columns] = data.loc[data.index, data.columns]

                data = temp_data
                data.loc[:, 'full_context_sequence_padded'] = left_pad + data.full_context_sequence + right_pad
            #Handle bystander_edit file
            elif f.startswith('bystander_edit'):
                cols = [c for c in data.columns if not c.startswith('Position')]
                new_cols = cols + pos_columns
                temp_data = pd.DataFrame(columns=new_cols, index=data.index)
                temp_data.loc[data.index,data.columns] = data.loc[data.index, data.columns]

                data = temp_data
                data.loc[:, 'full_context_sequence_padded'] = left_pad + data.full_context_sequence + right_pad

            # Drop genomic context and reorder columns
            des_cols = list(data.columns)

            #Remove certain columns
            for rm in remove_cols:
                if rm in des_cols:
                    des_cols.remove(rm)
            data = data.loc[:, des_cols]

            #Save data
            data.reset_index(inplace=True, drop=True)
            data.to_csv(f.split(f'_{latest_file_suffix}.csv')[0] + '_' + suffix_savings + '.csv')

            print(f"Saved {author} - {f} - harmonized and padding added")

# Add PAM columns and grna/seq mismatch flag
def add_pam_col_grna_seq_mismatch(authors, col_ordered, data_path, task='Bystander', latest_file_suffix="vLocation", suffix_savings="vF"):
    """
    Add PAM columns and grna/seq mismatch flag
    @params:
        authors                              - Required  : authors/studies to run (list)
        col_ordered                          - Required  : ordered columns for reordering columns in data files (list)
        data_path                            - Required  : file path to the processed data files (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
        latest_file_suffix                   - Required  : suffix of file version to use (str)
        suffix_savings                       - Required  : suffix that should be used for saving the file (str)
    """
    #Loop through all authors/studies
    for author in authors:
        #Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        file_names = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Loop through all files
        for f in file_names:
            #Read data
            data = pd.read_csv(f, index_col=0)
            cols = list(data.columns)

            data['pam_sequence'] = pd.NA
            data['grna_sequence_match'] = pd.NA

            #Add pam column and sequence mismatches
            for ind in data.index:
                seq = data.loc[ind, 'full_context_sequence']
                grna = data.loc[ind, 'grna']
                pam_index = data.loc[ind, 'pam_index']
                pam_seq = seq[pam_index:][:3]
                grna_seq_match = seq.find(grna[1:]) != -1

                data.loc[ind, 'pam_sequence'] = pam_seq
                data.loc[ind, 'grna_sequence_match'] = grna_seq_match

            # Order columns add PAM sequence and grna sequence match - pam after grna, grna_sequence_match after pam_index
            cols.insert(cols.index('pam_index')+1, 'grna_sequence_match')
            cols.insert(cols.index('grna')+1, 'pam_sequence')
            data = data.loc[:, cols]

            #Order data frame
            for c in col_ordered:
                if c not in cols:
                    data[c] = pd.NA

            col_ord = col_ordered + [c for c in data.columns if c.startswith('Position')]
            data = data.loc[:, col_ord]

            # Save data
            data.reset_index(inplace=True, drop=True)
            data.to_csv(f.split(f'_{latest_file_suffix}.csv')[0] + '_' + suffix_savings + '.csv')

            print(f"Saved {author} - {f} - added PAM and sequence grna mismatch")

# Add CRISPR spec energies to data file
def add_crispr_spec_energies(authors, data_path, crispr_spec_data_path, crispr_spec_file_name, task='Bystander', latest_file_suffix="vF", suffix_savings="vF"):
    """
    Add CRISPR spec energies to data file
    @params:
        authors                              - Required  : authors/studies to run (list)
        data_path                            - Required  : file path to the processed data files (str)
        crispr_spec_data_path                - Required  : file path to the CRISPRspec data files (str)
        crispr_spec_file_name                - Required  : file name of the CRISPRspec data files (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
        latest_file_suffix                   - Required  : suffix of file version to use (str)
        suffix_savings                       - Required  : suffix that should be used for saving the file (str)
    """
    # Set working directory
    os.chdir(crispr_spec_data_path)

    #Load data
    crispr_spec = pd.read_csv(crispr_spec_file_name, index_col=0)
    energy_cols = [c for c in crispr_spec.columns if c.startswith('energy')]

    #Loop through all authors/studies
    for author in authors:
        # Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        file_names = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Loop through all files
        for f in file_names:
            data = pd.read_csv(f, index_col=0)
            data[energy_cols] = pd.NA

            #Add crispr spec energies
            for ind in data.index:
                grna = data.loc[ind, 'grna']
                proto_index = data.loc[ind, 'protospace_position']
                target = data.loc[ind,'full_context_sequence'][proto_index:][:20]
                temp_crispr_spec = crispr_spec.loc[(crispr_spec.grna == grna) & (crispr_spec.target == target), :]

                if len(temp_crispr_spec) > 0:
                    data.loc[ind, energy_cols] = temp_crispr_spec.loc[:,energy_cols].values[0]

            # Save data
            data.reset_index(inplace=True, drop=True)
            data.to_csv(f.split(f'_{latest_file_suffix}.csv')[0] + '_' + suffix_savings + '.csv')

            print(f"Saved {author} - {f} - added CRISPR Off Energies")

# Add SCREEN data to data file
def add_screen_data(authors, new_cols, fields_to_replace, data_path, screen_data_path, screen_retrieval_step_size=10000000, task='Bystander', latest_file_suffix="vF", suffix_savings="vF"):
    """
    Add SCREEN data to data file
    @params:
        authors                              - Required  : authors/studies to run (list)
        new_cols                             - Required  : new columns to be added with screen data (list)
        fields_to_replace                    - Required  : column fields to replace with another name (list)
        data_path                            - Required  : file path to the processed data files (str)
        screen_data_path                     - Required  : file path to the SCREEN data files (str)
        screen_retrieval_step_size           - Required  : screen retrieval step size - used in file name (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
        latest_file_suffix                   - Required  : suffix of file version to use (str)
        suffix_savings                       - Required  : suffix that should be used for saving the file (str)
    """
    #Loop through all authors/studies
    for author in authors:
        # Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        file_names = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Loop through all files
        for f in file_names:
            #Load data
            data = pd.read_csv(f, index_col=0)
            data[new_cols] = pd.NA

            #Set working directory
            os.chdir(screen_data_path)

            #Subset data by assembly
            data_subset = data.loc[data.assembly == 'hg38'].copy()

            #Loop through all included chromosomes
            for chrom in data_subset.chromosome.unique():

                #Generate file name
                file_name_screen = f"GRCh38_{chrom}_{int(screen_retrieval_step_size)}.csv"

                if file_name_screen in os.listdir():
                    #Load screen data
                    screen = pd.read_csv(file_name_screen, index_col=0)
                    screen['end'] = screen.start + screen.len
                    screen_cols = list(screen.columns)
                    screen_cols[screen_cols.index('info.accession')] = 'accession_code_screen'
                    screen.columns = screen_cols

                    #Get all data frame indices that have a specific chromosome
                    ind_chrom = data_subset.loc[(data_subset.chromosome == chrom)].index
                    
                    #Loop through all 
                    for ind in ind_chrom:
                        start_edit = data_subset.loc[ind, 'location']
                        end_edit = start_edit + len(data_subset.loc[ind, 'sequence'])

                        #Match sequence locations with screen data
                        temp_screen = screen.loc[((screen.start >= start_edit) & (screen.end <= end_edit)) | ((screen.end >= start_edit) & (screen.start <= start_edit)) | ((screen.start <= start_edit) & (screen.end >= end_edit)) | ((screen.end >= end_edit) & (screen.start <= end_edit))].copy()

                        #Check fit if multiple matches
                        if len(temp_screen) > 0:
                            # Select best match based on max overlap and smallest sequences
                            edit_range = list(range(int(start_edit), int(end_edit)))
                            temp_screen['percentage_bases_covered'] = pd.NA

                            for t_ind in temp_screen.index:
                                range_screen = list(range(int(temp_screen.loc[t_ind, 'start']), int(temp_screen.loc[t_ind, 'end'])))
                                overlap = len(list(set(edit_range).intersection(range_screen)))
                                temp_screen.loc[t_ind, 'percentage_bases_covered'] = overlap / (end_edit - start_edit)

                            temp_screen.sort_values(['percentage_bases_covered', 'len'], ascending=[False, True], inplace=True)
                            data_subset.loc[ind,new_cols] = temp_screen.loc[temp_screen.index[0], new_cols]

            # Save to main df
            data.loc[data_subset.index,new_cols] = data_subset.loc[data_subset.index, new_cols]
            data.index = data.original_id

            # Adjust col names for promoter and enhancer
            cur_cols = list(data.columns)
            for old_field, new_field in fields_to_replace:
                print([old_field, new_field])
                col_index = cur_cols.index(old_field)
                if col_index != -1:
                    cur_cols[col_index] = new_field
            data.columns = cur_cols

            #Save data
            data.reset_index(inplace=True, drop=True)
            os.chdir(data_path + '\\' + author + '\\' + task)
            data.to_csv(f.split(f'_{latest_file_suffix}.csv')[0] + '_' + suffix_savings + '.csv')

            print(f"Saved {author} - {f} - added SCREEN data")

# Add RNAfold MFE to data file
def add_RNAfold_energy(authors, data_path, RNAfold_data_path, RNAfold_file_name, task='Bystander', latest_file_suffix="vF", suffix_savings="vF"):
    """
    Add RNAfold MFE to data file
    @params:
        authors                              - Required  : authors/studies to run (list)
        data_path                            - Required  : file path to the processed data files (str)
        RNAfold_data_path                    - Required  : file path to the RNAfold data files (str)
        RNAfold_file_name                    - Required  : file name of the RNAfold data file (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
        latest_file_suffix                   - Required  : suffix of file version to use (str)
        suffix_savings                       - Required  : suffix that should be used for saving the file (str)
    """
    #Set working directory
    os.chdir(RNAfold_data_path)

    #Load data
    rna_fold = pd.read_csv(RNAfold_file_name, index_col=1)
    rna_fold.index = [r.replace('U', 'T') for r in rna_fold.index]
    
    energy_cols = ['free_energy']
    
    #Loop through all authors/studies
    for author in authors:
        # Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        file_names = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Add energy terms to all files
        for f in file_names:
            data = pd.read_csv(f, index_col=0)

            data.index = data.grna

            data[energy_cols] = pd.NA
            common_ind = list(set(data.index).intersection(set(rna_fold.index)))

            for ind in common_ind:
                data.loc[ind, energy_cols] = rna_fold.loc[ind,energy_cols].values[0]

            # Save data
            data.reset_index(inplace=True, drop=True)
            data.to_csv(f.split(f'_{latest_file_suffix}.csv')[0] + '_' + suffix_savings + '.csv')

            print(f"Saved {author} - {f} - added RNAfold energy")

# Add melting temperature to data files
def add_melting_temperature(authors, data_path, melting_temp_data_path, melting_temp_file_name, task='Bystander', latest_file_suffix="vF", suffix_savings="vF"):
    """
    Add melting temperature to data files
    @params:
        authors                              - Required  : authors/studies to run (list)
        data_path                            - Required  : file path to the processed data files (str)
        melting_temp_data_path               - Required  : file path to the melting temperature data files (str)
        melting_temp_file_name               - Required  : file name of the melting temperature data file (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
        latest_file_suffix                   - Required  : suffix of file version to use (str)
        suffix_savings                       - Required  : suffix that should be used for saving the file (str)
    """
    #Set working directory
    os.chdir(melting_temp_data_path)

    #Load data
    temp_df = pd.read_csv(melting_temp_file_name, index_col=0)
    temp_df.drop_duplicates(inplace=True)
    temp_cols = ['melt_temperature_grna', 'melt_temperature_target']

    #Loop through all authors/studies
    for author in authors:
        # Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        file_names = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Add temperatures for all data files
        for f in file_names:
            #Read data
            data = pd.read_csv(f, index_col=0)
            data[temp_cols] = pd.NA

            #gRNA  temperatures
            data.index = data.grna
            temp_df.index = temp_df.grna
            common_ind = list(set(data.index).intersection(temp_df.index))

            #Assign gRNA melting temperature values
            for ind in common_ind:
                if sum(temp_df.index == ind) > 1:
                    data.loc[ind, 'melt_temperature_grna'] = temp_df.loc[ind,'melt_temperature_grna'].values[0]
                else:
                    data.loc[ind, 'melt_temperature_grna'] = temp_df.loc[ind,'melt_temperature_grna']

            #Target temperatures
            data.index = data.original_id
            vals = []

            #Compute target sequence
            for ind in data.index:
                seq = data.loc[ind, 'full_context_sequence']
                proto_index = data.loc[ind, 'protospace_position']

                vals.append(seq[proto_index:][:20])

            data.index = vals
            temp_df.index = temp_df.target
            common_ind = list(set(data.index).intersection(temp_df.index))

            #Assign target melting temperature values
            for ind in common_ind:
                if sum(temp_df.index == ind) > 1:
                    data.loc[ind, 'melt_temperature_target'] = temp_df.loc[ind,'melt_temperature_target'].values[0]
                else:
                    data.loc[ind, 'melt_temperature_target'] = temp_df.loc[ind,'melt_temperature_target']

            # Save data
            data.reset_index(inplace=True, drop=True)
            data.to_csv(f.split(f'_{latest_file_suffix}.csv')[0] + '_' + suffix_savings + '.csv')

            print(f"Saved {author} - {f} - added melting temperature")

# Merge all author data files into one file
def merge_data_files(authors, author_ids, rel_cols_outcome, rel_cols_edit, data_path, savings_path, task='Bystander', latest_file_suffix="vF"):
    """
    Merge all author data files into one file
    @params:
        authors                              - Required  : authors/studies to run (list)
        author_ids                           - Required  : author/study id (dict)
        rel_cols_outcome                     - Required  : relevant columns for the outcome bystander view (list)
        rel_cols_edit                        - Required  : relevant columns for the edit bystander view (list)
        data_path                            - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save data files (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
        latest_file_suffix                   - Required  : suffix of file version to use (str)
        suffix_savings                       - Required  : suffix that should be used for saving the file (str)
    """
    #Create empty data frame
    df = pd.DataFrame()

    #Loop through all authors/studies
    for author in authors:
        # Set working directory
        os.chdir(data_path + '\\' + author + '\\' + task)

        file_name = [f for f in os.listdir() if f.endswith(f'{latest_file_suffix}.csv')]

        #Read data
        data = pd.read_csv(file_name[0], index_col=0)

        #Add study id
        data['study_id'] = author_ids[author]

        #Column handling to extract right columns
        cols = list(data.columns)
        if rel_cols_outcome[0] not in list(data.columns):
            col_insert = [c for c in data.columns if c.startswith('Position_')][-1]
            data[rel_cols_outcome] = pd.NA
            cols = cols[:cols.index(col_insert)+1] + list(rel_cols_outcome) + cols[cols.index(col_insert)+1:]

        if rel_cols_edit[0] not in list(data.columns):
            col_insert = [c for c in data.columns if c.startswith('Position_')][0]
            data[rel_cols_edit] = pd.NA
            cols = cols[:cols.index(col_insert)] + list(rel_cols_edit) + cols[cols.index(col_insert):]

        if len(file_name) > 1:
            # Load second file and add edit/outcome position data
            data_alt = pd.read_csv(file_name[1], index_col=0)
            rel_cols = [c for c in data_alt.columns if c.startswith('Position_')]
            data.loc[data_alt.index, rel_cols] = data_alt.loc[:, rel_cols]

        #Concate data
        data = data.loc[:, cols]
        data.index = list(range(len(df), len(df)+len(data)))
        df = pd.concat([df, data], axis=0)

    #Save data
    os.chdir(savings_path)
    df.reset_index(inplace=True, drop=True)
    df.to_csv('data_vF.csv')

#Calculate stats for individual studies
def get_study_data_stats(authors, processed_data_path, savings_path, task = 'Bystander'):
    """
    Calculate stats for individual studies
    @params:
        authors                              - Required  : authors/studies to run (list)
        processed_data_path                  - Required  : file path to the processed data files (str)
        savings_path                         - Required  : file path to save stats (str)
        task                                 - Required  : machine learning task - used in the folder structure (str)
    """
    #Create empty data frame
    cols = ['guides', 'targets', 'cells', 'base_editors']
    df = pd.DataFrame(index=authors, columns=cols)

    #Loop through all authors/studies
    for author in authors:
        # Set working directory
        os.chdir(processed_data_path + '\\' + author + '\\' + task)

        #Read data
        file_names = [f for f in os.listdir() if f.startswith('bystander') and f.endswith('vF.csv')]
        bystander_file_name = file_names[0]
        data = pd.read_csv(bystander_file_name, index_col=0)

        #Compute stats
        df.loc[author, 'targets'] = len(data.sequence.unique())
        df.loc[author, 'guides'] = len(data.grna.unique())
        cells = list(data.cell.unique())
        cells.sort()
        df.loc[author, 'cells'] = str(cells).replace("\'", "").replace("[", "").replace("]", "")
        base_editors = list(data.base_editor.unique())
        base_editors.sort()
        df.loc[author, 'base_editors'] = str(base_editors).replace("\'", "").replace("[", "").replace("]", "")

    os.chdir(savings_path)
    df.to_csv('study_stats_overview.csv')

#Calculate stats for the final data file
def get_data_stats(file_name, stats_fields, processed_data_path, savings_path, group_by = 'base_editor'):
    """
    Calculate stats for the final data file
    @params:
        file_name                            - Required  : file name of the data file (str)
        stats_fields                         - Required  : list of dictionary outlining the fields and stats to be calculated - example: [{'field':'location','type':'count_unique'}] (list(dict))
        processed_data_path                  - Required  : file path to the final merged data files (str)
        savings_path                         - Required  : file path to save stats (str)
        group_by                             - Required  : column data should be grouped by for stats (str)
    """
    #Create empty data frame
    df = pd.DataFrame()

    #Read data
    os.chdir(processed_data_path)
    data = pd.read_csv(file_name, index_col = 0)

    #Change index to group by
    data.index = data.loc[:, group_by]

    #Loop through unique indices and stats fields
    for ind in data.index.unique():
        for sf in stats_fields:
            field = sf['field']
            stat_type = sf['type']

            #Subset data
            subset_data = data.loc[ind,field].dropna(how='all',axis=0)

            #Calculate stats
            if stat_type == 'count_unique':
                stat_res = len(subset_data.unique())
            elif stat_type == 'count':
                stat_res = len(subset_data)
            elif stat_type == 'count_inverse':
                total_length = len(subset_data)
                stat_res = total_length - sum(subset_data.values == True)
            elif stat_type == 'average':
                stat_res = subset_data.values.mean()
            elif stat_type == 'unique':
                stat_res = str(subset_data.unique())
            elif stat_type == 'count_any':
                stat_res = len(subset_data.dropna(how='all',axis=0))
                
            #Assign values
            field_save = str(field) + "_" + stat_type
            df.loc[ind,field_save] = stat_res
    
    #Save data
    os.chdir(savings_path)
    df.to_csv(f'stats_overview_{group_by}.csv')
