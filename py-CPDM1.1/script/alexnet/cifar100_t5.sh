export CUDA_VISIBLE_DEVICES="0"
SEED=0
export PYTHONHASHSEED=$SEED

python run_CPDM.py \
    --seed $SEED \
    --class_incremental \
    --method_name CPDM \
    --model_name alexnetCI_pretrained \
    --ds_name cifar100CI \
    --batch_size 256 \
    --num_epochs 100 \
    --drop_margin 0.5 \
    --max_attempts_per_task 1 \
    --test_set test \
    --test_overwrite_mode \
    --max_task_count 6 \
    --test_max_task_count 6 \
    --CI_task_count 5 \
    --label_hid_dim 768 \
    --image_size 64 \
    --num_samples 100 \
    --CPDM_generator_factor 0.25 \
    --use_fp16 True \
    --diffusion_steps 1000 \
    --num_inference_timesteps 100 \
    --diffusion_lr 1e-4 \
    --diffusion_batch_size 64 \
    --diffusion_lr_warmup_steps 500 \
    --lr_anneal_steps 15000 \
    --log_interval 10 \
    --save_interval 500 \
    --class_cond True \
    --drop_labels_prob 0.2 \
    --w 3.0 \
    --image_condition learn \
    --image_condition_learn_lr 1e-2 \
    --image_condition_learn_L2 1e-2 \
    --max_grad_norm \
    --nearest_label cosine \
    --tau 1e-4 \
    --CI_order_rndseed 12551 \