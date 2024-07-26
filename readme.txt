# for train
(aloha) yoshikawa@rtx01:~/job/2024/airec/ACT/act$ python3 imitate_episodes.py \
 --task_name sportStacking --ckpt_dir ./log \
 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 4 \
 --dim_feedforward 3200 --num_epochs 10000  --lr 1e-5 --seed 0

# for test
(aloha) yoshikawa@rtx01:~/job/2024/airec/ACT/act$ python3 test.py \
--ckpt_dir ./log/<tag> --chunk_size 100 --policy_class ACT --task_name sportStacking \
--batch_size 3 --seed 0 --num_epochs 10000 --lr 1e-2 --kl_weight 10  --hidden_dim 512 \
--dim_feedforward 3200



### model 20240726_0227_24 or later uses images_raw.npy