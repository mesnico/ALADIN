#!/bin/bash

#-------------- prepare some paths (do not touch unless you change installation directories) -------------- #

SG_BENCHMARK_PATH=/media/nicola/Data/Workspace/OSCAR/scene_graph_benchmark
ALADIN_PATH=/media/nicola/Data/Workspace/OSCAR/Oscar

#-------------- prepare some useful variables (modify if needed) -------------- #

DATASET=$1

if [[ "$DATASET" == "V3C1" ]]; then
    IMG_PATH=/media/datino/Dataset/VBS/V3C_dataset/V3C1
    OUT_PATH=/media/nicola/SSD/VBS_Features/V3C1_ALADIN
    IMG_LIST_FILE=${IMG_PATH}/v3c1_image_list.txt
elif [[ "$DATASET" == "V3C2" ]]; then
    IMG_PATH=/media/datino/Dataset/VBS/V3C_dataset/V3C2
    OUT_PATH=/media/nicola/SSD/VBS_Features/V3C2_ALADIN
    IMG_LIST_FILE=${IMG_PATH}/v3c2_image_list.txt
else
    echo "Dataset ${DATASET} not recognized!"
    exit 1;
fi

BATCH_SIZE=10000 # how many images we extract the feature of before running ALADIN and dumping to h5

mkdir -p $OUT_PATH
TOTAL_FILES=$(wc -l < $IMG_LIST_FILE)
echo "Found ${TOTAL_FILES} images."

for FROM in $(seq 0 $BATCH_SIZE $TOTAL_FILES)
do

    sleep 1

    TO=$(( $FROM + $BATCH_SIZE ))
    TO=$(( $TO > $TOTAL_FILES ? $TOTAL_FILES : $TO ))
    OUT_H5_FILE=${OUT_PATH}/aladin_features_${FROM}_${TO}.h5

    if [ -f "$OUT_H5_FILE" ]; then
        echo "File $OUT_H5_FILE already existing. Skipping..."
        continue
    fi

    #-------------- Run object detector to get visual features --------------#

    cd $SG_BENCHMARK_PATH

    PYTHONPATH=. conda run --no-capture-output -n sg_benchmark python extraction_service/test_sg_net.py \
    --config-file sgg_configs/vgattr/vinvl_x152c4.yaml \
    --img_folder $IMG_PATH \
    --file_list_txt $IMG_LIST_FILE \
    --from_idx $FROM \
    --to_idx $TO \
    TEST.IMS_PER_BATCH 4 \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    DATA_DIR temp_tsv_folder \
    TEST.IGNORE_BOX_REGRESSION True \
    MODEL.ATTRIBUTE_ON True \
    TEST.OUTPUT_FEATURE True \
    DATASETS.LABELMAP_FILE models/vinvl/VG-SGG-dicts-vgoi6-clipped.json \
    DATASETS.TEST "(\"train.yaml\", )" \
    DATALOADER.NUM_WORKERS 4

    #-------------- Extract ALADIN features (only if the previous extraction phase completed successfully) --------------#

    if [ $? -ne 0 ]; then
        echo "Error during the run of object detection. Skipping ALADIN extraction."
        continue 
    fi

    cd $ALADIN_PATH

    #conda run --no-capture-output -n oscar python 
    PYTHONPATH=. conda run --no-capture-output -n oscar python alad/extraction/extract_visual_features.py\
    --data_dir /media/nicola/SSD/OSCAR_Datasets/coco_ir\
    --img_feat_file ${SG_BENCHMARK_PATH}/output/X152C5_test/inference/vinvl_vg_x152c4/predictions.tsv\
    --eval_model_dir /media/nicola/SSD/OSCAR_Datasets/checkpoint-0132780\
    --max_seq_length 50\
    --max_img_seq_length 34\
    --load_checkpoint alad/runs/alad-alignment-and-distill/model_best_rsum.pth.tar\
    --features_h5 $OUT_H5_FILE

done