
# convert autra dataset to openscene dataset


# generate feature_fusion
#export CUDA_VISIBLE_DEVICES=2,3
#python3.8 scripts/feature_fusion/autra_ovseg_multi_thread.py   --data_dir /home/fan.ling/big_model/OpenScene/OpenScene/data  --output_dir /home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test   --split train

# 3d distill
export CUDA_VISIBLE_DEVICES=4,5
#sh run/distill.sh lingfan_autra_ovseg_distill_120 config/nuscenes/ours_ovseg_pretrained_autra_data.yaml
#sh run/distill.sh lingfan_autra_ovseg_distill_2075 config/nuscenes/ours_ovseg_pretrained_autra_data.yaml

# eval
#export CUDA_VISIBLE_DEVICES=6
#sh run/eval.sh out/mask2former_2075 config/nuscenes/ours_ovseg_pretrained_autra_data.yaml distill
