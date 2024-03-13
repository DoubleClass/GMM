#!/bin/bash  
for((i=1;i<=1;i++));  
do   
    # echo $i
    TXT_PATH='/home/ubuntu/code/GMM_camera/GMM/results/imgr20_20_3e6_batch2_epoch2_order3_2000exp_re/test_'"$i"'.txt'
    ton-gpfs-caoxusheng/code/mini-CIL/minigpt4/output/tiny_100_5_batch2_epoch1_2_3e6_7_order3/20231112152/20/checkpoint_1.pth
    CKPT_PATH='/home/ubuntu/code/GMM_camera/GMM/minigpt4/output/imgr20_20_3e6_batch2_epoch2_order3_2000exp_re/20240312095/'"$i"'/checkpoint_4.pth'

    python batch_eval.py --cfg-path eval_configs/minigpt4_eval_all_tasks_imgr.yaml \
    --gpu-id 0 --task-id $i --txt-path $TXT_PATH \
    --ckpt-path $CKPT_PATH
done  
python get_score_all.py