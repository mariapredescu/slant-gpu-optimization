#!/bin/bash

PIECES_LIST=("1_1_1" "1_1_2" "1_1_3" "1_2_1" "1_2_2" "1_2_3" "1_3_1" "1_3_2" "1_3_3" "2_1_1" "2_1_2" "2_1_3" "2_2_1" "2_2_2" "2_2_3" "2_3_1" "2_3_2" "2_3_3" "3_1_1" "3_1_2" "3_1_3" "3_2_1" "3_2_2" "3_2_3" "3_3_1" "3_3_2" "3_3_3")

PIECE="3_3_1"
GPU_ID=0
GET_USED_MEMORY="nvidia-smi --query-gpu=memory.usage --format=csv -i $GPU_ID | grep -Eo [0-9]+"
GET_FREE_MEMORY="nvidia-smi --query-gpu=memory.free --format=csv -i $GPU_ID | grep -Eo [0-9]+"
FRAGMENTS_LIST=()

# compose the list of fragmets to process for each image
for image in "$@"
do
        for PIECE in "${PIECES_LIST[@]}"
        do
                echo "$image:$PIECE"
                FRAGMENTS_LIST+=("$image:$PIECE")
        done
done

used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv -i $GPU_ID | grep -Eo [0-9]+)
echo $used_memory

for fragment in "${FRAGMENTS_LIST[@]}"
do
        P_IMG=$(echo $fragment | cut -d':' -f 1)
        P_FRAG=$(echo $fragment | cut -d':' -f 2)
        echo $P_IMG
        echo $P_FRAG

        $PWD/pythondir/miniconda3/bin/python $PWD/extra/python/test.py --piece=$P_FRAG --model_dir=$PWD/extra/model_dir --test_img_dir=$PWD/OUTPUTS/$P_IMG/working_dir/deep_learning --out_dir=$PWD/OUTPUTS/$P_IMG/working_dir/all_pieces &
        sleep 1

        used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv -i $GPU_ID | grep -Eo [0-9]+)
        free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv -i $GPU_ID | grep -Eo [0-9]+)
        while [ $free_memory -lt 6000 ]
        do
                free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv -i $GPU_ID | grep -Eo [0-9]+)
                sleep 1
                echo "$free_memory in loop"
        done
done