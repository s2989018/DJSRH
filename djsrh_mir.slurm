#!/bin/bash
#SBATCH --job-name=djsrh_mir
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="yesminaelarkoubi@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge

module load ALICE/default
module load slurm
# module load CUDA/12.3.2
# module load cuDNN/8.9.7.29
# module load GCCcore/12.2.0
module load CUDA/12.3.2
module load GCC/11.3.0
module load Python/3.10.8-GCCcore-12.2.0

# Source the Python virtual environment
source $HOME/cirh_dir/CIRH/test_venv/bin/activate


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES
nvidia-smi

OUT_DIR="./datasets/MIRFlickr_djsrh"

if [ -d "$OUT_DIR" ]; then
    echo "[$SHELL] ## Preprocessing already exists. Skipping."
else
    echo "[$SHELL] ## Running preprocessing..."
    python preprocessing.py
fi

echo "[$SHELL] ## Starting DJSRH training"
python train.py
