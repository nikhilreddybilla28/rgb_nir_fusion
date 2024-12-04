#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --job-name=internimage_idde
#SBATCH --output=log_outputs/idde_internimage_b_fog_%J.out

source activate internimage
module load u18/cuda/10.2
# module load u18/cudnn/8.3.3-cuda-10.2

scratch_dir="/ssd_scratch/nikhil.reddy"
mkdir -p ${scratch_dir}

work_dir="work_dirs"
share3_dir=nikhil.reddy@ada:/share3/nikhil.reddy

if [ ! -f "${scratch_dir}/IDDAW_final" ]; then
	# Loading data from dataset to scratch
	# rsync -a nikhil.reddy@ada:/share1/dataset/cityscapes  ${scratch_dir}/
	rsync -avz nikhil.reddy@ada:/share3/nikhil.reddy/Datasets/Final_Dataset/IDDAW_final ${scratch_dir}/
	# mkdir -p ${work_dir}
    # unzip IDD.zip
	
fi

# Running the script
#export PYTHONPATH=$PYTHONPATH:${pruners_lib_dir}
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
WANDB_START_METHOD="thread"
# python semseg.py --dataset=${scratch_dir}/cityscapes train -c 19 --arch=${arch} --epochs=${epochs}  --lr=${lr} --batch-size=${batch_size} | tee nosparsity_train.txt
cd ~/InternImage/segmentation

CONFIG="configs/idd_aw/upernet_internimage_b_iddaw_fog.py"
# GPUS=2
# PORT=${PORT:-24050}
# CHECKPOINT='pretrained_models/internimage_cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.pth'
CHECKPOINT="work_dirs/upernet_internimage_b_512x1024_160k_idd/best_mIoU_iter_320000.pth"

python train.py $CONFIG --load-from $CHECKPOINT --work_dir work_dirs/upernet_internimage_b_iddaw_fog


# cd ${scratch_dir}
# zip -r work_dir.zip $work_dir
# rsync -avz $work_dir $share3_dir
