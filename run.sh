export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=8d739e5eaa28091db300de37eb709020ff7cf27c
# python /home/tsinghuaair/zhengkx/conservative-world-models/main_offline.py vcfb quadruped rnd --eval_tasks escape fetch jump roll_fast roll stand
python /home/tsinghuaair/zhengkx/conservative-world-models/main_offline.py vcfb walker rnd --eval_tasks run stand flip walk 
