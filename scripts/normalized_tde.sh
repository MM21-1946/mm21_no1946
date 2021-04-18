# find all configs in configs/
models=(
    2dtan/activitynet/baseline_causal
    2dtan/charades/baseline_causal
    cmin/activitynet/baseline_causal
    cmin/charades/baseline_causal
    csmgan/activitynet/baseline_causal
    csmgan/charades/baseline_causal
    fian/activitynet/baseline_causal
    fian/charades/baseline_causal
    lgi/activitynet/baseline_causal
    lgi/charades/baseline_causal
)
# set your gpu id
gpus=0
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi task on the same machine
master_addr=127.0.0.1
master_port=29501

# ------------------------ need not change -----------------------------------
for model in ${models[*]}; do
    config_file=configs/$model\.yaml
    output_dir=outputs/$model
    
    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
    --nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port train_net.py --config-file $config_file --skip-test OUTPUT_DIR $output_dir
done

