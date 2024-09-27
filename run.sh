export CUDA_VISIBLE_DEVICES=0
# export WANDB_API_KEY=183745c61e3c0db51eb85b3fcda31b527854b7aa
python main_offline.py iexp walker rnd \
--eval_tasks run stand flip walk \
# --wandb_logging False \
--seed 42 \
--z_dimension 50 \
--weighted_cml False \
--dataset_transitions 100000 \
--learning_steps 1000000 \
--cql_alpha 0.01 \
--alpha 0.1 \
--target_conservative_penalty 50 \
--lagrange True