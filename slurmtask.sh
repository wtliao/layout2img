#!/bin/bash

#SBATCH --mail-user=liao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=graph # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=./slurm_log/train_graph_context_app.txt         # Logdatei für den merged STDOUT/STDERR output
#SBATCH --partition=gpu_cluster_enife # Partition auf der gerechnet werden soll
                                    #   ohne Angabe des Parameters wird auf der Default-Partition gerechnet
#SBATCH --time=1-0                  # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS oder Tage-Stunden)
#SBATCH --nodes=1                   # Reservierung von 2 Rechenknoten
                                    #   alle nachfolgend reservierten CPUs müssen sich auf den reservierten Knoten befinden
#SBATCH --gres=gpu:6
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
echo "Here begin to train_graph_context_app"
srun hostname
source activate pyt
python train_graph_context_app.py
# python IS.py
echo "training  train_graph_context_app!"
