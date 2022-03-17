## Installation
### Requirements
- Python 3.7
- Pytorch 1.2
- torchvision 0.4.0
- cuda 10.0

### Setup with Conda
```bash
# create a new environment
conda create --name oscar python=3.7
conda activate oscar

# install pytorch1.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

export INSTALL_DIR=$PWD

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
python setup.py install --cuda_ext --cpp_ext

# install this repo
cd $INSTALL_DIR
git clone --recursive https://github.com/mesnico/OSCAR-TERAN-distillation
cd OSCAR-TERAN-distillation/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install requirements
pip install -r requirements.txt

unset INSTALL_DIR
```

### Download OSCAR & Vin-VL Retrieval data:
Download the checkpoint folder with [azcopy](https://docs.microsoft.com/it-it/azure/storage/common/storage-use-azcopy-v10):
```
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/coco_ir/base/checkpoint-0132780/' <checkpoint-target-folder> --recursive
```

Download the IR data
```
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_ir' <data-folder> --recursive
```

Download the pre-extracted Bottom-Up features 
```
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/' <features-folder> --recursive
```

## Training:
``` 
cd teran 

python train_finetune.py --do_test --do_eval --num_captions_per_img_val 5 --data_dir <data-folder>/coco_ir --img_feat_file <features-folder>/features.tsv --cross_image_eval --per_gpu_eval_batch_size 64 --eval_model_dir <checkpoint-target-folder>/checkpoint-0132780 --config configs/teran_finetune_on_oscar_coco.yaml --logger_name <output-folder> --val_step 7000 --max_seq_length 50 --max_img_seq_length 34
```
