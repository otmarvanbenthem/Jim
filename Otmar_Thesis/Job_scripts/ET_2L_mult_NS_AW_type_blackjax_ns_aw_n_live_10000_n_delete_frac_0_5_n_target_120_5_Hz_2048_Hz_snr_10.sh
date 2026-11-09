#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --mem=30G
#SBATCH -t 40:00:00
#SBATCH --output=slurm-ET_2L_mult_NS_AW_type_blackjax_ns_aw_n_live_10000_n_delete_frac_0_5_n_target_120_5_Hz_2048_Hz_snr_10.out

# Silence the JAX cache warning
export PYTHONWARNINGS="ignore:Error writing persistent compilation cache entry:UserWarning"

source $HOME/Jim/.venv/bin/activate
cd $HOME/Jim/Otmar_Thesis/

start=$SECONDS
jim-run /home/otmar-van-benthem/Documents/Jim/Otmar_Thesis/Job_scripts/ET_2L_mult_NS_AW_type_blackjax_ns_aw_n_live_10000_n_delete_frac_0_5_n_target_120_5_Hz_2048_Hz_snr_10.toml
runtime=$((SECONDS - start))
echo "Python step took $runtime s"
mkdir -p /home/ovbenthem/DATA/ET_2L_mult_NS_AW_type_blackjax_ns_aw_n_live_10000_n_delete_frac_0_5_n_target_120_5_Hz_2048_Hz_snr_10
echo "$runtime" > /home/ovbenthem/DATA/ET_2L_mult_NS_AW_type_blackjax_ns_aw_n_live_10000_n_delete_frac_0_5_n_target_120_5_Hz_2048_Hz_snr_10/runtime_seconds.txt
