#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --job-name=internimage_idde
#SBATCH --output=log_outputs/idde_internimage_b_%J.out
#SBATCH --mail-user=furqan.shaik@research.iiit.ac.in

source activate internimage
module load u18/cuda/10.2
# module load u18/cudnn/8.3.3-cuda-10.2

scratch_dir="/ssd_scratch/cvit/furqan.shaik"
mkdir -p ${scratch_dir}
cd ${scratch_dir}
mkdir idd_aw
cd idd_aw
mkdir -p train
mkdir -p test
cd train
mkdir -p leftImg8bit
# mkdir -p csTrainIds
mkdir -p level3Ids
# mkdir -p nir
# mkdir -p stacked
mkdir -p gtFine
cd ..
cd test
mkdir -p leftImg8bit
# mkdir -p csTrainIds
mkdir -p level3Ids
# mkdir -p nir
# mkdir -p stacked
mkdir -p gtFine

work_dir="work_dirs"
share3_dir=furqan.shaik@ada:/share3/furqan.shaik

if [ ! -f "${scratch_dir}/IDDAW_final" ]; then
	# Loading data from dataset to scratch
	# rsync -a furqan.shaik@ada:/share1/dataset/cityscapes  ${scratch_dir}/
	rsync -avz furqan.shaik@ada:/share3/furqan.shaik/Datasets/Final_Dataset/IDDAW_final ${scratch_dir}/
	
	cd ${scratch_dir}
	rsync -avz IDDAW_final/train/*/level3Ids idd_aw/train/
	rsync -avz IDDAW_final/train/*/leftImg8bit idd_aw/train/
	# rsync -avz IDDAW_final/train/*/csTrainIds idd_aw/train/
	# rsync -avz IDDAW_final/train/*/nir idd_aw/train/
	# rsync -avz IDDAW_final/train/*/stacked idd_aw/train/
	rsync -avz IDDAW_final/train/*/gtFine idd_aw/train/

	rsync -avz IDDAW_final/test/*/level3Ids idd_aw/test/
	rsync -avz IDDAW_final/test/*/leftImg8bit idd_aw/test/
	# rsync -avz IDDAW_final/test/*/csTrainIds idd_aw/test/
	# rsync -avz IDDAW_final/test/*/nir idd_aw/test/
	# rsync -avz IDDAW_final/test/*/stacked idd_aw/test/
	rsync -avz IDDAW_final/test/*/gtFine idd_aw/test/
	# mkdir -p ${work_dir}
    # unzip IDD.zip
	
fi

# Running the script
#export PYTHONPATH=$PYTHONPATH:${pruners_lib_dir}
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
WANDB_START_METHOD="thread"
# python semseg.py --dataset=${scratch_dir}/cityscapes train -c 19 --arch=${arch} --epochs=${epochs}  --lr=${lr} --batch-size=${batch_size} | tee nosparsity_train.txt
cd ~/InternImage/segmentation

CONFIG="configs/idd_aw/upernet_internimage_b_iddaw.py"
# GPUS=2
# PORT=${PORT:-24050}
# CHECKPOINT='pretrained_models/internimage_cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.pth'
CHECKPOINT="work_dirs/upernet_internimage_b_iddaw_resume/best_mIoU_iter_128000.pth"

python train.py $CONFIG --work_dir work_dirs/upernet_internimage_b_iddaw_heirloss_nopretrain

# cd ${scratch_dir}
# zip -r work_dir.zip $work_dir
# rsync -avz $work_dir $share3_dir