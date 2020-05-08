#!/bin/bash

#SBATCH -N 1                                                    # number of nodes (due to the nature of sequential processing, here u$
#SBATCH -n 20                                                    # number of cores (here uses 2)
#SBATCH --time=20:30:00                                       # time allocation, which has the format
#SBATCH --job-name face_rgbd_gpu ##name that will show up in the queue
#SBATCH --output face_rgbd_gpu.o%j ##filename of the output; the "%j" will append the jobID to the end of the name making the output file$
#SBATCH --mem=60G


#Notification configuration
#SBATCH --gres=K80
#SBATCH --mail-type BEGIN
#SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which me$
#SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which m$
#SBATCH --mail-user=naimaotberd@gmail.com                       # Email to which notification will be sent


# load modules
module load R/3.3.1
python combine_parts.py
