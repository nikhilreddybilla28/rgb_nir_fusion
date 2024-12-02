# source activate internimage
# module load u18/cuda/10.2
# # module load u18/cudnn/8.3.3-cuda-10.2

# scratch_dir="/ssd_scratch/cvit/furqan.shaik"
# mkdir -p ${scratch_dir}
# # work_dir="work_dirs/resume"
# # share3_dir=furqan.shaik@ada:/share3/furqan.shaik

# if [ ! -f "${scratch_dir}/cityscapes" ]; then
# 	# Loading data from dataset to scratch
# 	# rsync -a furqan.shaik@ada:/share1/dataset/cityscapes  ${scratch_dir}/
# 	rsync -avz furqan.shaik@ada:/share1/dataset/cityscapes ${scratch_dir}/
#     cd ${scratch_dir}
# 	cd cityscapes/
# 	# mkdir -p ${work_dir}
#     tar -xvf cityscapes.tar
	
# fi

# # Running the script
# #export PYTHONPATH=$PYTHONPATH:${pruners_lib_dir}
# #export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# # python semseg.py --dataset=${scratch_dir}/cityscapes train -c 19 --arch=${arch} --epochs=${epochs}  --lr=${lr} --batch-size=${batch_size} | tee nosparsity_train.txt
# cd ~/InternImage/segmentation

# # CONFIG="configs/idd/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes.py"
# # GPUS=2
# # # PORT=${PORT:-24050}
# # CHECKPOINT=work_dirs/segformer_internimage_xl_512x1024_160k_mapillary2cityscapes/iter_32000.pth

# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# # python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
# #     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# # torchrun train.py $CONFIG --resume-from $CHECKPOINT --gpus 2

python test.py configs/cityscapes/upernet_internimage_s_512x1024_160k_cityscapes.py \
	pretrained_models/internimage_cityscapes/upernet_internimage_s_512x1024_160k_cityscapes.pth \
	--eval mIoU

python test.py configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py \
	pretrained_models/internimage_cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.pth \
	--eval mIoU

python test.py configs/cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.py \
	pretrained_models/internimage_cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.pth \
	--eval mIoU

python test.py configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py \
	pretrained_models/internimage_cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.pth \
	--eval mIoU

python test.py configs/cityscapes/upernet_internimage_xl_512x1024_160k_cityscapes.py \
	pretrained_models/internimage_cityscapes/upernet_internimage_xl_512x1024_160k_cityscapes.pth \
	--eval mIoU

python test.py configs/cityscapes/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.py \
	pretrained_models/internimage_cityscapes/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.pth \
	--eval mIoU

python test.py configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py \
	pretrained_models/internimage_cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth \
	--eval mIoU