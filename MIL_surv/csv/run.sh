#!/bin/bash
#SBATCH --job-name=survival_excel
#SBATCH --output=./outputs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_short
#SBATCH --mem=18000

# Activate anaconda environment code
module load anaconda3/2021.05/gcc-9.2.0
source activate $HOME/.conda/envs/s4

python surv_csv.py --path_files /gpfs/workdir/chrakii/FEATURES_HES/RESNET50/pt \
                        --path_clinical_csv /gpfs/workdir/chrakii/RRT-MIL/Survival/csv/PEACE1_scanned_clinical_data_14042023_IHC_full.csv \
                        --path_metadata hes_os.csv \
                        --event_col delosrandopointy \
                        --study os \
                        --status_col osstatuspointv   


python surv_csv.py --path_files /gpfs/workdir/chrakii/FEATURES_HES/RESNET50/pt \
                        --path_clinical_csv /gpfs/workdir/chrakii/RRT-MIL/Survival/csv/PEACE1_scanned_clinical_data_14042023_IHC_full.csv \
                        --path_metadata hes_rpfs.csv \
                        --event_col delrpfsrandopointy \
                        --study rpfs \
                        --status_col rpfsstatuspointv   


# delrpfsrandopointy delosrandopointy osstatuspointv rpfsstatuspointv