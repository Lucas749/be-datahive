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
import os

#Import own functions
from initial_data_processing import parse_marquart_data, parse_pallaseni_data, parse_song_data, parse_yuan_data, process_arbab_data, flatten_arbab_data
from data_enrichment import run_cas_offinder,add_cas_offinder_results,match_location_via_biopython,get_encode_data,run_RNA_fold,unpack_RNA_fold_output,run_CRISPR_spec,calculate_melting_temperatures,harmonize_data_padding,add_pam_col_grna_seq_mismatch,add_crispr_spec_energies,add_screen_data,add_RNAfold_energy,add_melting_temperature,merge_data_files,get_study_data_stats, get_data_stats

#######################################################
# Run Initial Data Processing
#######################################################
#Variables
file_path = r'<RAW_DATA_PATH>'
saving_path = r'<SAVING_DATA_PATH>'
editing_window_start = 3
editing_window_end = 10

############
#Marquart
############
author = 'Marquart'
task = 'Bystander'
file_path_ext = '\\' + author + '\\' + task

file_name = "bystander.xlsx"
columns = ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count',
           'edited_count', 'efficiency_full_grna_calculated', 'error_flag', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated']
num_positions_before_protospacer = 0
num_positions = 19

print(f'Running {author}...')
parse_marquart_data(file_name, editing_window_start, editing_window_end, num_positions_before_protospacer, num_positions, columns, file_path, saving_path, file_path_ext)

############
#Arbab
############
author = 'Arbab'
task = 'Bystander'
file_path_ext = '\\' + author + '\\' + task
eff_data_path = r'<ARBAB_EFFICIENCY_DATA_PATH>'
file_name = 'bystander.csv'
num_positions_before_protospacer = 9
num_positions = 30

#Unwrap raw data
print(f'Running {author} raw data...')
columns = ['original_id', 'grna', 'sequence', 'outcome', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'total_count_aggregated',
        'edited_count_eff', 'edited_count_bys', 'edited_count_outcome', 'fraction_edited_outcome', 'fraction_edited_eff', 'fraction_edited_bys', 'error_flag', 'duplicated_key', 'file_name']
flatten_arbab_data(columns, file_path, eff_data_path, saving_path, file_path_ext)

#Calculate data file
print(f'Running {author} data file...')
columns = ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor', 'total_count', 'edited_count', 'efficiency_full_grna_calculated',
        'error_flag', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated', 'efficiency_data_total_count', 'efficiency_data_edited_count', 'efficiency_data_efficiency_full_grna','efficiency_full_grna_reported', 'total_count_reported_efficiency', 'edited_count_reported_efficiency']
process_arbab_data(file_name, editing_window_start, editing_window_end, num_positions_before_protospacer, num_positions, columns, file_path, saving_path, file_path_ext)

############
#Pallaseni
############
author = 'Pallaseni'
task = 'Bystander'
file_path_ext = '\\' + author + '\\' + task
oligo_data_path = file_path + '\\' + author + '\\' + 'Oligo_Data_Pallaseni.csv'
abe_efficiency_file_name = 'total_efficiency_values.raw.abe.tsv'
cbe_efficiency_file_name = 'total_efficiency_values.raw.cbe.tsv'
eff_data_path = file_path + '\\' + author + '\\Efficiency'
file_name = 'forecast_be_processed_data.csv' 
columns = ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor',
        'total_count_reported_efficiency', 'edited_count', 'efficiency_full_grna_calculated', 'efficiency_full_grna_reported', 'error_flag', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated']
num_positions_before_protospacer = 9
num_positions = 30

print(f'Running {author}...')
parse_pallaseni_data(file_name, abe_efficiency_file_name, cbe_efficiency_file_name, num_positions_before_protospacer, num_positions, columns, file_path, eff_data_path, oligo_data_path, saving_path, file_path_ext)

############
#Yuan
############
author = 'Yuan'
task = 'Bystander'
file_path_ext = '\\' + author + '\\' + task
oligo_data_path = file_path + '\\' + author + '\\' + 'Oligo_Data.csv'
file_name = 'bystander.csv'
columns = ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor',
        'total_count', 'edited_count', 'efficiency_full_grna', 'error_flag', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated']
num_positions = 20

print(f'Running {author}...')
parse_yuan_data(file_name, num_positions, columns, file_path, oligo_data_path, saving_path, file_path_ext)

############
#Song
############
author = 'Song'
task = 'Bystander'
file_path_ext = '\\' + author + '\\' + task
file_name = 'Supplementary table 2_MyungjaeSong2_readable.xlsx'

columns = ['original_id', 'grna', 'sequence', 'full_context_sequence', 'protospace_position', 'pam_index', 'genomic_context', 'genomic_context_build', 'cell', 'base_editor',
           'total_count', 'edited_count', 'efficiency_full_grna_calculated', 'error_flag', 'scaling_factor_edited_count_denominator', f'editing_windows_{editing_window_start}_{editing_window_end}_efficiency_calculated', 'editing_windows_3_10_efficiency_reported','total_count_reported_efficiency','edited_count_reported_efficiency']
num_positions_before_protospacer = 4
num_positions = 25

print(f'Running {author}...')
parse_song_data(file_name, editing_window_start, editing_window_end, num_positions_before_protospacer, num_positions, columns, file_path, saving_path, file_path_ext, rebase_outcome_proportions = True)

#######################################################
# Run Data Enrichment
#######################################################
############
#Location
############
authors = ["Pallaseni", "Yuan", "Song", "Marquart"]
max_mismatches = 10
max_mismatches_ratio = 0.5
assembly_path = './hg38.chromFa/chroms'
cas_offinder_path = r'<CAS-OFFINDER_PATH>'
savings_path = r'<CAS-OFFINDER_SAVING_PATH>'
processed_data_path = r"<SAVING_DATA_PATH>"
task = 'Bystander'

run_cas_offinder(authors, max_mismatches, max_mismatches_ratio, assembly_path, processed_data_path, savings_path, cas_offinder_path, task)

# Add casoffinder results
authors = ["Pallaseni", "Song", "Yuan", "Marquart"]
location_cols = ['assembly', 'chromosome', 'location', 'direction','mismatches', 'assembly_sequence', 'not_unique_match_flag']
processed_data_path = r"<SAVING_DATA_PATH>"
savings_path = r'<CAS-OFFINDER_SAVING_PATH>' 
task = 'Bystander'

add_cas_offinder_results(authors, location_cols, assembly_path, processed_data_path, savings_path, task)

# BioPython Matching Sequence - Arbab
author = "Arbab"
assembly_path = r'<CAS-OFFINDER_PATH>'
processed_data_path = r"<SAVING_DATA_PATH>"
savings_path = processed_data_path
task = 'Bystander'

match_location_via_biopython(author, assembly_path, processed_data_path, savings_path, task)

############
#SCREEN
############
# Download all SCREEN data
savings_path = r'<SCREEN_PATH>'
assembly = 'GRCh38'
step_size = 10e6
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

os.chdir(savings_path)
#Loop through all chromosomes
for chr in chromosomes:
    data = get_encode_data(chr, assembly=assembly,step_size=step_size, max_bp_length=500e6)

    data['duplicates'] = data.start.astype(str) + "_" + data.enhancer_zscore.astype(str) + "_" + data.ctcf_zscore.astype(str) + "_" + data.dnase_zscore.astype(str) + "_" + data.pct.astype(str)
    data.drop_duplicates(subset=['duplicates'], inplace=True)

    data.to_csv(f"{assembly}_{chr}_{int(step_size)}.csv")

############
#Energies
############
#RNAfold
RNA_fold_path = r"<VIENNE_RNA_PATH>"
savings_path = r'<RNA_FOLD_SAVING_PATH>'
processed_data_path = r"<SAVING_DATA_PATH>"
task = 'Bystander'
authors = ["Pallaseni", "Yuan", "Song", "Marquart", "Arbab"]

run_RNA_fold(authors, processed_data_path, savings_path, RNA_fold_path)

#Unpack RNA fold output into a data frame
file_path = fr'{savings_path}\output.txt'
unpack_RNA_fold_output(file_path, savings_path)

#CRISPRspec
authors = ["Pallaseni", "Yuan", "Song", "Marquart", "Arbab"]
param_values = [[True, False]]*6
rel_comb = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
processed_data_path = r"<SAVING_DATA_PATH>"
savings_path = r'<CRISPR_OFF_SAVING_PATH>'
crispr_spec_path = fr'{savings_path}\crisproff'
task = 'Bystander'

#Run CRISPRspec to calculate free energy terms
run_CRISPR_spec(authors, param_values, rel_comb, processed_data_path, savings_path, crispr_spec_path, task)

############
#Mt Temps
############
# Authors to check
authors = ["Pallaseni", "Yuan", "Song", "Marquart", "Arbab"]
savings_path = r'<MELT_TEMP_SAVING_PATH>'
processed_data_path = r"<SAVING_DATA_PATH>"
task = 'Bystander'

#Calculate melting temperatures
calculate_melting_temperatures(authors, processed_data_path, savings_path, task)

############
#Add Data
############
data_path = r"<SAVING_DATA_PATH>"
save_path = data_path + '\\merged'

#Authors
authors = ["Arbab", "Pallaseni", "Yuan", "Song", "Marquart"]

# Harmonize data and add padding to the target sequence
remove_cols = ['genomic_context', 'genomic_context_build','scaling_factor_edited_count_denominator', 'error_flag']
harmonize_data_padding(authors, remove_cols, data_path, grna_length=20, task='Bystander', latest_file_suffix="vLocation", suffix_savings="vF")
    
#PAM col and grna seq mistmatch
col_ordered = ["original_id", "grna", "pam_sequence", "sequence", "full_context_sequence", "full_context_sequence_padded", "protospace_position", "pam_index", "grna_sequence_match", "assembly", "chromosome", "location", "direction", "mismatches", "assembly_sequence", "not_unique_match_flag",
                "cell", "base_editor", "total_count_reported_efficiency", "edited_count_reported_efficiency", "total_count", "edited_count", "efficiency_full_grna_reported", "editing_windows_3_10_efficiency_reported", "efficiency_full_grna_calculated", "editing_windows_3_10_efficiency_calculated"]
add_pam_col_grna_seq_mismatch(authors, col_ordered, data_path, latest_file_suffix="vF", suffix_savings="vF")

#Add screen data
screen_data_path = r'<SCREEN_PATH>'
new_cols = ['ctcf_zscore', 'dnase_zscore', 'enhancer_zscore', 'promoter_zscore', 'percentage_bases_covered', 'accession_code_screen']
fields_to_replace = [['enhancer_zscore', 'H3K27ac_zscore'], ['promoter_zscore', 'H3K4me3_zscore']]
add_screen_data(authors, new_cols, fields_to_replace, data_path, screen_data_path, screen_retrieval_step_size=10000000,task='Bystander', latest_file_suffix="vF", suffix_savings="vF")

#Add crispr spec
crispr_spec_data_path = r'<CRISPR_OFF_SAVING_PATH>'
crispr_spec_file_name = 'crisproff_energies_all_combinations.csv'
add_crispr_spec_energies(authors, data_path, crispr_spec_data_path, crispr_spec_file_name,task='Bystander', latest_file_suffix="vF", suffix_savings="vF")

#Add RNAfold energy
RNAfold_data_path = r'<RNA_FOLD_SAVING_PATH>'
RNAfold_file_name = 'RNAfold_output.csv'
add_RNAfold_energy(authors, data_path, RNAfold_data_path, RNAfold_file_name, task='Bystander', latest_file_suffix="vF", suffix_savings="vF")

#Add melting temperature
melting_temp_data_path = r'<MELT_TEMP_SAVING_PATH>'
melting_temp_file_name = 'melting_temperature.csv'
add_melting_temperature(authors, data_path, melting_temp_data_path, melting_temp_file_name, task='Bystander', latest_file_suffix="vF", suffix_savings="vF")

#Merge files
author_ids = {"Arbab": 1, "Yuan": 2, "Pallaseni": 3, "Marquart": 4, "Song": 5}
rel_cols_outcome = ['Position_-11 A', 'Position_-11 T', 'Position_-11 C', 'Position_-11 G', 'Position_-10 A', 'Position_-10 T', 'Position_-10 C', 'Position_-10 G', 'Position_-9 A', 'Position_-9 T', 'Position_-9 C', 'Position_-9 G', 'Position_-8 A', 'Position_-8 T', 'Position_-8 C', 'Position_-8 G', 'Position_-7 A', 'Position_-7 T', 'Position_-7 C', 'Position_-7 G', 'Position_-6 A', 'Position_-6 T', 'Position_-6 C', 'Position_-6 G', 'Position_-5 A', 'Position_-5 T', 'Position_-5 C', 'Position_-5 G', 'Position_-4 A', 'Position_-4 T', 'Position_-1 A', 'Position_-1 T', 'Position_-1 C', 'Position_-1 G', 'Position_0 A', 'Position_0 T', 'Position_0 C', 'Position_0 G', 'Position_1 A', 'Position_1 T', 'Position_1 C', 'Position_4 C', 'Position_4 G', 'Position_5 A', 'Position_5 T', 'Position_5 C', 'Position_5 G', 'Position_6 A', 'Position_6 T', 'Position_6 C', 'Position_6 G', 'Position_7 A', 'Position_7 T', 'Position_7 C', 'Position_7 G', 'Position_8 A', 'Position_8 T', 'Position_8 C', 'Position_8 G', 'Position_9 A', 'Position_9 T', 'Position_9 C', 'Position_9 G', 'Position_10 A', 'Position_10 T', 'Position_10 C', 'Position_10 G', 'Position_11 A', 'Position_11 T', 'Position_11 C', 'Position_11 G', 'Position_12 A', 'Position_12 T', 'Position_12 C',
                    'Position_12 G', 'Position_13 A', 'Position_13 T', 'Position_13 C', 'Position_13 G', 'Position_14 A', 'Position_14 T', 'Position_14 C', 'Position_14 G', 'Position_15 A', 'Position_15 T', 'Position_15 C', 'Position_15 G', 'Position_16 A', 'Position_16 T', 'Position_16 C', 'Position_16 G', 'Position_17 A', 'Position_17 T', 'Position_17 C', 'Position_17 G', 'Position_18 A', 'Position_18 T', 'Position_18 C', 'Position_18 G', 'Position_19 A', 'Position_19 T', 'Position_19 C', 'Position_19 G', 'Position_20 A', 'Position_20 T', 'Position_20 C', 'Position_20 G', 'Position_21 A', 'Position_21 T', 'Position_21 C', 'Position_21 G', 'Position_22 A', 'Position_22 T', 'Position_22 C', 'Position_22 G', 'Position_23 A', 'Position_23 T', 'Position_23 C', 'Position_23 G', 'Position_24 A', 'Position_24 T', 'Position_24 C', 'Position_24 G', 'Position_25 A', 'Position_25 T', 'Position_25 C', 'Position_25 G', 'Position_26 A', 'Position_26 T', 'Position_26 C', 'Position_26 G', 'Position_27 A', 'Position_27 T', 'Position_27 C', 'Position_27 G', 'Position_28 A', 'Position_28 T', 'Position_28 C', 'Position_28 G', 'Position_29 A', 'Position_29 T', 'Position_29 C', 'Position_29 G', 'Position_30 A', 'Position_30 T', 'Position_30 C', 'Position_30 G']
rel_cols_edit = ['Position_-11', 'Position_-10', 'Position_-9', 'Position_-8', 'Position_-7', 'Position_-6', 'Position_-5', 'Position_-4', 'Position_-3', 'Position_-2', 'Position_-1', 'Position_0', 'Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7', 'Position_8', 'Position_9',
                    'Position_10', 'Position_11', 'Position_12', 'Position_13', 'Position_14', 'Position_15', 'Position_16', 'Position_17', 'Position_18', 'Position_19', 'Position_20', 'Position_21', 'Position_22', 'Position_23', 'Position_24', 'Position_25', 'Position_26', 'Position_27', 'Position_28', 'Position_29', 'Position_30']


merge_data_files(authors, author_ids, rel_cols_outcome, rel_cols_edit, data_path, save_path, task='Bystander', latest_file_suffix="vF")

############
#Get Stats
############
processed_data_path = r"<SAVING_DATA_PATH>"
savings_path = processed_data_path + '\\merged'

#By Authors
authors = ["Arbab", "Pallaseni", "Yuan", "Song", "Marquart"]
get_study_data_stats(authors, processed_data_path, savings_path, )

#By base editor
file_name = 'data_vF.csv'
processed_data_path = fr"{data_path}\merged"
stats_fields = [{'field':'location','type':'count'},
{'field':'grna','type':'count_unique'},
{'field':'sequence','type':'count_unique'},
{'field':'mismatches','type':'average'},
{'field':'not_unique_match_flag','type':'count_inverse'},
{'field':'assembly','type':'unique'},
{'field':'efficiency_full_grna_reported','type':'average'},
{'field':['ctcf_zscore','H3K27ac_zscore','H3K4me3_zscore'],'type':'count_any'},
]
get_data_stats(file_name, stats_fields, processed_data_path, savings_path, group_by = 'base_editor')