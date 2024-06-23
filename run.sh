export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=8d739e5eaa28091db300de37eb709020ff7cf27c
python main_offline.py cexp walker rnd \
--eval_tasks run stand flip walk \
--wandb_logging True \
--seed 42 \
--z_dimension 50 \
--weighted_cml False \
--dataset_transitions 100000 \
--learning_steps 1000000 \
--cql_alpha 0.01 \
--alpha 0.1 \
--target_conservative_penalty 50 \
--lagrange True