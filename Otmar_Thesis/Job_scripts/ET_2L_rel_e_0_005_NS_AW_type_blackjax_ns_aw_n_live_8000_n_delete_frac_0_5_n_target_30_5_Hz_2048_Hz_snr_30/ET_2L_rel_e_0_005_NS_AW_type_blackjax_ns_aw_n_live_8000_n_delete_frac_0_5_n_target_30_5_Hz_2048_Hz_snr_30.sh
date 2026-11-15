#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --mem=30G
#SBATCH -t 40:00:00
#SBATCH --output=slurm-ET_2L_rel_e_0_005_NS_AW_type_blackjax_ns_aw_n_live_8000_n_delete_frac_0_5_n_target_30_5_Hz_2048_Hz_snr_30.out

# Silence the JAX cache warning
export PYTHONWARNINGS="ignore:Error writing persistent compilation cache entry:UserWarning"
 
source $HOME/Jim/.venv/bin/activate
cd $HOME/Jim/Otmar_Thesis/
 
start=$SECONDS
jim-run $HOME/Jim/Otmar_Thesis/Job_scripts/ET_2L_rel_e_0_005_NS_AW_type_blackjax_ns_aw_n_live_8000_n_delete_frac_0_5_n_target_30_5_Hz_2048_Hz_snr_30/ET_2L_rel_e_0_005_NS_AW_type_blackjax_ns_aw_n_live_8000_n_delete_frac_0_5_n_target_30_5_Hz_2048_Hz_snr_30.toml
runtime=$((SECONDS - start))
echo "Python step took $runtime s"
mkdir -p /gpfs/home3/ovbenthem/DATA/ET_2L_rel_e_0_005_NS_AW_type_blackjax_ns_aw_n_live_8000_n_delete_frac_0_5_n_target_30_5_Hz_2048_Hz_snr_30
echo "$runtime" > /gpfs/home3/ovbenthem/DATA/ET_2L_rel_e_0_005_NS_AW_type_blackjax_ns_aw_n_live_8000_n_delete_frac_0_5_n_target_30_5_Hz_2048_Hz_snr_30/runtime_seconds.txt
