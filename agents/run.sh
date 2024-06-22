export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=8d739e5eaa28091db300de37eb709020ff7cf27c
python /home/tsinghuaair/zhengkx/conservative-world-models/main_offline.py cexp walker rnd --eval_tasks stand walk flip run
# python /home/tsinghuaair/zhengkx/conservative-world-models/main_offline.py iexp walker rnd --eval_tasks stand walk flip run
# python /home/tsinghuaair/zhengkx/conservative-world-models/main_offline.py ifb walker rnd --eval_tasks stand walk flip run
