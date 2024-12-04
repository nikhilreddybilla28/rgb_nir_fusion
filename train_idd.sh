#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 19
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --job-name=internimage_idd
#SBATCH --output=log_outputs/idd_internimage_xl_%J.out
#SBATCH --mail-user=nikhil.reddy@research.iiit.ac.in

source activate internimage
module load u18/cuda/10.2
# module load u18/cudnn/8.3.3-cuda-10.2

scratch_dir="/ssd_scratch/cvit/nikhil.reddy"
mkdir -p ${scratch_dir}
work_dir="work_dirs/resume"
share3_dir=nikhil.reddy@ada:/share3/nikhil.reddy

if [ ! -f "${scratch_dir}/IDD_Segmentation" ]; then
	# Loading data from dataset to scratch
	# rsync -a nikhil.reddy@ada:/share1/dataset/cityscapes  ${scratch_dir}/
	rsync -avz nikhil.reddy@ada:/share3/nikhil.reddy/Datasets/IDD.zip ${scratch_dir}/
    cd ${scratch_dir}
	# mkdir -p ${work_dir}
    unzip IDD.zip
	
fi

# Running the script
#export PYTHONPATH=$PYTHONPATH:${pruners_lib_dir}
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# python semseg.py --dataset=${scratch_dir}/cityscapes train -c 19 --arch=${arch} --epochs=${epochs}  --lr=${lr} --batch-size=${batch_size} | tee nosparsity_train.txt
cd ~/InternImage/segmentation

CONFIG="configs/idd/upernet_internimage_b_512x1024_160k_idd.py"
GPUS=2
# PORT=${PORT:-24050}
pretrained_model="pretrained_models/internimage_cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.pth"
# CHECKPOINT=work_dirs/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes/iter_32000.pth

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
torchrun train.py $CONFIG --resume-from $CHECKPOINT --gpus 2

# cd ${scratch_dir}
# zip -r work_dir.zip $work_dir
# rsync -avz $work_dir $share3_dir
