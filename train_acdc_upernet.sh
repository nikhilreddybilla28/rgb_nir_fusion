#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --job-name=internimage_acdc
#SBATCH --output=log_outputs/acdc_internimage_b_%J.out
#SBATCH --mail-user=furqan.shaik@research.iiit.ac.in

source activate internimage
module load u18/cuda/10.2
# module load u18/cudnn/8.3.3-cuda-10.2

scratch_dir="/ssd_scratch/cvit/furqan.shaik"
mkdir -p ${scratch_dir}
mkdir -p ${scratch_dir}/data
cd ${scratch_dir}
cd data
mkdir -p acdc
cd acdc
mkdir -p rgb_anon
mkdir -p gt
cd rgb_anon
mkdir -p train
mkdir -p val
cd ../gt
mkdir -p train 
mkdir -p val

work_dir="work_dirs"
share3_dir=furqan.shaik@ada:/share3/furqan.shaik

if [ ! -f "${scratch_dir}/ACDC" ]; then
	# Loading data from dataset to scratch
	# rsync -a furqan.shaik@ada:/share1/dataset/cityscapes  ${scratch_dir}/
	rsync -avz furqan.shaik@ada:/share3/furqan.shaik/Datasets/ACDC ${scratch_dir}/
	
	cd ${scratch_dir}/ACDC
	unzip gt_trainval.zip
	unzip rgb_anon_trainvaltest.zip
	cd ..
	rsync -a ACDC/rgb_anon/*/train/*/* data/acdc/rgb_anon/train/
	rsync -a ACDC/rgb_anon/*/val/*/* data/acdc/rgb_anon/val/
	rsync -a ACDC/gt/*/train/*/*_labelTrainIds.png data/acdc/gt/train/
	rsync -a ACDC/gt/*/val/*/*_labelTrainIds.png data/acdc/gt/val/
	# mkdir -p ${work_dir}
    # unzip IDD.zip
	
fi

# Running the script
#export PYTHONPATH=$PYTHONPATH:${pruners_lib_dir}
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
WANDB_START_METHOD="thread"
# python semseg.py --dataset=${scratch_dir}/cityscapes train -c 19 --arch=${arch} --epochs=${epochs}  --lr=${lr} --batch-size=${batch_size} | tee nosparsity_train.txt
cd ~/InternImage/segmentation

CONFIG="configs/acdc/upernet_internimage_b_512x1024_160k_acdc.py"
# GPUS=2
# PORT=${PORT:-24050}
# CHECKPOINT="pretrained_models/internimage_cityscapes/upernet_internimage_s_512x1024_160k_cityscapes.pth"
CHECKPOINT=pretrained_models/internimage_cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.pth

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
torchrun train.py $CONFIG --load-from $CHECKPOINT --gpus 2

# cd ${scratch_dir}
# zip -r work_dir.zip $work_dir
# rsync -avz $work_dir $share3_dir