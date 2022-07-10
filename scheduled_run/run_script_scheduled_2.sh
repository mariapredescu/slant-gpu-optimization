#!/bin/bash

PIECES_LIST=("1_1_1" "1_1_2" "1_1_3" "1_2_1" "1_2_2" "1_2_3" "1_3_1" "1_3_2" "1_3_3" "2_1_1" "2_1_2" "2_1_3" "2_2_1" "2_2_2" "2_2_3" "2_3_1" "2_3_2" "2_3_3" "3_1_1" "3_1_2" "3_1_3" "3_2_1" "3_2_2" "3_2_3" "3_3_1" "3_3_2" "3_3_3")

PIECE="3_3_1"

echo "script usage: ./script.sh -i <gpu-id> -t <time-interval> -n <no-of-images> -- <list-of-images>"

while [ -n "$1" ]
do
        case "$1" in
                -i) GPU_ID=$2 # the gpu id where the images are processed
                        shift;;
                -t) TIME_INTERVAL=$2 # time interval to wait before starting another batch of pieces
                        shift;;
                -n) NO_OF_IMAGES=$2 # number of images to be processed (should be smaller than the max number of segments)
                        shift;;
                --) shift
                        break;;
                *) echo "Option $1 not recognized";;
        esac
        shift
done

total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv -i $GPU_ID | grep -Eo [0-9]+)

max_no_of_segments=$(($total_mem/5000))

if [ $max_no_of_segments -gt 5 ]
then
        max_no_of_segments=5
fi

if [ $max_no_of_segments -lt $NO_OF_IMAGES ]
then
        echo "warning: The maximum number of images that can be processed at a time for this GPU config is $max_no_of_segments. Enter a number of images smaller than that."
        exit 0
fi

free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $GPU_ID | grep -Eo [0-9]+)
echo "gpu id $GPU_ID"
echo "time interval $TIME_INTERVAL"
echo "number of images $NO_OF_IMAGES"
echo $total_mem
echo $free_mem

for PIECE in "${PIECES_LIST[@]}"
do
        echo "image segment $PIECE"
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $GPU_ID | grep -Eo [0-9]+)
        while [ $free_mem -lt 6000 ]
        do
                free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $GPU_ID | grep -Eo [0-9]+)
                sleep 1
                echo "$free_mem in loop"
        done
        echo "$free_mem after loop"

        for image in $@
        do
                $PWD/pythondir/miniconda3/bin/python $PWD/extra/python/test.py --piece=$PIECE --model_dir=$PWD/extra/model_dir --test_img_dir=$PWD/OUTPUTS/$image/working_dir/deep_learning --out_dir=$PWD/OUTPUTS/$image/working_dir/all_piece &
                echo "process fragment $PIECE of image $image"
        done
        sleep $TIME_INTERVAL
done