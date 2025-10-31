accelerate launch \
    --multi_gpu meta_train_deepspeed.py \
    --config-name Qwen3-0.6B data.train_batch_size=1 \
    data.eval_batch_size=2 \
    run.gradient_accumulation_steps=4 \
    mode=pretrain \
    data.source=transmla