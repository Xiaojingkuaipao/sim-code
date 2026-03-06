#!/usr/bin/env zsh

# 创建日志目录，避免输出丢失
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"

echo "Starting All Experiments... Logs will be saved to ${LOG_FILE}" | tee -a ${LOG_FILE}

# 固定的全局参数
SEQ_LEN=8192
EMBED_SIZE=8192
GPUS_PER_NODE=8

# 定义执行单个实验的辅助函数
run_experiment() {
    local exp_name=$1
    local num_nodes=$2
    local batch_size=$3
    local topk=$4
    local zipf=$5

    echo "\n========================================================================" | tee -a ${LOG_FILE}
    echo ">>> Running Experiment: ${exp_name}" | tee -a ${LOG_FILE}
    echo ">>> Config: ${num_nodes} Nodes | Batch: ${batch_size} | TopK: ${topk} | Zipf: ${zipf}" | tee -a ${LOG_FILE}
    echo "========================================================================" | tee -a ${LOG_FILE}

    # 将参数放入数组中（Zsh语法）
    local args=(
        --num_nodes ${num_nodes}
        --gpus_per_node ${GPUS_PER_NODE}
        --seq_len ${SEQ_LEN}
        --embed_size ${EMBED_SIZE}
        --batch_size ${batch_size}
        --topk ${topk}
        --zipf_alpha ${zipf}
        --workload_output_dir "./workload"
    )

    echo "\n[1/4] Generating Workload..." | tee -a ${LOG_FILE}
    python generate_workload.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo "\n[2/4] Running DeepEP Baseline..." | tee -a ${LOG_FILE}
    python deepEP.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo "\n[3/4] Running FAST Baseline..." | tee -a ${LOG_FILE}
    python FAST.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo "\n[4/4] Running Ours..." | tee -a ${LOG_FILE}
    python ours-simulator-fine.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo "\n<<< Completed Experiment: ${exp_name}" | tee -a ${LOG_FILE}
}

# ==============================================================================
# 📍 图 7: 不同负载强度下的性能 (Varying Message Size)
# ==============================================================================

# 图 7(a): 8x8 集群, zipf=0.5, topk=2
echo "\n\n⭐⭐⭐ Starting Figure 7(a) Experiments ⭐⭐⭐" | tee -a ${LOG_FILE}
batches_7a=(64 128 256 512)
sizes_7a=("128M" "256M" "512M" "1G")
for i in {1..4}; do
    run_experiment "Fig7a_${sizes_7a[$i]}" 8 ${batches_7a[$i]} 2 0.5
done

# 图 7(b): 16x8 集群, zipf=0.5, topk=2
echo "\n\n⭐⭐⭐ Starting Figure 7(b) Experiments ⭐⭐⭐" | tee -a ${LOG_FILE}
batches_7b=(128 256 512 1024)
sizes_7b=("128M" "256M" "512M" "1G")
for i in {1..4}; do
    run_experiment "Fig7b_${sizes_7b[$i]}" 16 ${batches_7b[$i]} 2 0.5
done

# ==============================================================================
# 📍 图 8: 不同偏斜度下的鲁棒性 (Varying Zipf Alpha)
# ==============================================================================
zipfs=(0.0 0.25 0.5 0.75 0.99)

# 图 8(a): 8x8 集群, 1G 负载 (Batch=512), topk=2
echo "\n\n⭐⭐⭐ Starting Figure 8(a) Experiments ⭐⭐⭐" | tee -a ${LOG_FILE}
for z in ${zipfs[@]}; do
    run_experiment "Fig8a_Zipf_${z}" 8 512 2 $z
done

# 图 8(b): 16x8 集群, 1G 负载 (Batch=1024), topk=2
echo "\n\n⭐⭐⭐ Starting Figure 8(b) Experiments ⭐⭐⭐" | tee -a ${LOG_FILE}
for z in ${zipfs[@]}; do
    run_experiment "Fig8b_Zipf_${z}" 16 1024 2 $z
done

# ==============================================================================
# 📍 图 9: 路由稀疏度适应性 (Varying Top-K)
# ==============================================================================
topks=(2 4 6 8)

# 图 9: 16x8 集群, 512M 负载 (Batch=512), zipf=0.5
echo "\n\n⭐⭐⭐ Starting Figure 9 Experiments ⭐⭐⭐" | tee -a ${LOG_FILE}
for k in ${topks[@]}; do
    run_experiment "Fig9_TopK_${k}" 16 512 $k 0.5
done

# ==============================================================================
echo "\n🎉 All Experiments Completed Successfully! 🎉" | tee -a ${LOG_FILE}
echo "Log saved at: ${LOG_FILE}" | tee -a ${LOG_FILE}