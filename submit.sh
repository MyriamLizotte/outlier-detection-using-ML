#!/bin/bash
#SBATCH --account=def-cgreenwo
#SBATCH --job-name='REPEN'
#SBATCH --mail-user=myriam.lizotte@mail.mcgill.ca 
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/errors_%j.err
#SBATCH --time=00:30:00 
#SBATCH --time=07:00:00 
#SBATCH --nodes=1
#SBATCH --mem=15G
#8SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
nvidia-smi

module load python/3.8.2
virtualenv --no-download $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate

pip install --no-index tensorflow
pip install --no-index scipy #==1.5.2
pip install --no-index scikit-learn
pip install --no-index matplotlib
pip install --no-index pandas #==1.1.3

cd $SLURM_SUBMIT_DIR
SRC=$SLURM_SUBMIT_DIR

python ./REPEN.py

