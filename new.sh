#!/bin/bash

# 创建日志目录，避免输出丢失
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"

echo "Starting All Experiments... Logs will be saved to ${LOG_FILE}" | tee -a ${LOG_FILE}

# 固定的全局参数
SEQ_LEN=8192
EMBED_SIZE=8192
GPUS_PER_NODE=8
DTYPE="float16"

# 定义执行单个实验的辅助函数
run_experiment() {
    local exp_name=$1
    local num_nodes=$2
    local batch_size=$3
    local topk=$4
    local zipf=$5
    local hot_rank_ratio=$6
    local hot_remote_min=$7
    local hot_remote_max=$8
    local cold_remote_min=$9
    local cold_remote_max=${10}

    echo -e "\n========================================================================" | tee -a ${LOG_FILE}
    echo ">>> Running Experiment: ${exp_name}" | tee -a ${LOG_FILE}
    echo ">>> Config: ${num_nodes} Nodes | Batch: ${batch_size} | TopK: ${topk} | Zipf: ${zipf}" | tee -a ${LOG_FILE}
    echo ">>> Skew: RankRatio=${hot_rank_ratio} | HotRemote=[${hot_remote_min}, ${hot_remote_max}] | ColdRemote=[${cold_remote_min}, ${cold_remote_max}]" | tee -a ${LOG_FILE}
    echo "========================================================================" | tee -a ${LOG_FILE}

    # 将参数放入数组中（Bash语法）
    local args=(
        --num_nodes ${num_nodes}
        --gpus_per_node ${GPUS_PER_NODE}
        --seq_len ${SEQ_LEN}
        --embed_size ${EMBED_SIZE}
        --batch_size ${batch_size}
        --dtype ${DTYPE}
        --topk ${topk}
        --zipf_alpha ${zipf}
        --hot_rank_ratio ${hot_rank_ratio}
        --hot_remote_ratio ${hot_remote_min} ${hot_remote_max}
        --cold_remote_ratio ${cold_remote_min} ${cold_remote_max}
        --workload_output_dir "./new_test_workload"
    )

    echo -e "\n[1/4] Generating Workload..." | tee -a ${LOG_FILE}
    # 使用 new-generate-workload.py
    python new-generate-workload.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo -e "\n[2/4] Running DeepEP Baseline..." | tee -a ${LOG_FILE}
    python deepEP.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo -e "\n[3/4] Running FAST Baseline..." | tee -a ${LOG_FILE}
    python fast.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo -e "\n[4/4] Running Ours..." | tee -a ${LOG_FILE}
    python ours-simulator-fine.py "${args[@]}" 2>&1 | tee -a ${LOG_FILE}

    echo -e "\n<<< Completed Experiment: ${exp_name}" | tee -a ${LOG_FILE}
}

# # ==============================================================================
# # 图 7: 不同负载强度下的性能 (Varying Message Size)
# # ==============================================================================

# # 固定参数
# ZIPF_7=0.25
# TOPK_7=6
# # Skew Config for Fig 7
# HRR_7=0.25
# HR_MIN_7=0.3
# HR_MAX_7=0.4
# CR_MIN_7=0.1
# CR_MAX_7=0.2

# # 图 7(a): 8x8 集群 (64 GPUs)
# echo -e "\n \n Starting Figure 7(a) Experiments (8x8 Nodes)" | tee -a ${LOG_FILE}
# # Message Size: 128M, 256M, 512M, 1G
# # Batch Size:   64,   128,  256,  512
# batches_7a=(64 128 256 512)
# sizes_7a=("128M" "256M" "512M" "1G")

# for i in {0..3}; do
#     run_experiment "Fig7a_${sizes_7a[$i]}" 8 ${batches_7a[$i]} ${TOPK_7} ${ZIPF_7} \
#                    ${HRR_7} ${HR_MIN_7} ${HR_MAX_7} ${CR_MIN_7} ${CR_MAX_7}
# done

# # 图 7(b): 16x8 集群 (128 GPUs)
# echo -e "\n \nStarting Figure 7(b) Experiments (16x8 Nodes) " | tee -a ${LOG_FILE}
# # Message Size: 128M, 256M, 512M, 1G
# # Batch Size:   128,  256,  512,  1024
# batches_7b=(128 256 512 1024)
# sizes_7b=("128M" "256M" "512M" "1G")

# for i in {0..3}; do
#     run_experiment "Fig7b_${sizes_7b[$i]}" 16 ${batches_7b[$i]} ${TOPK_7} ${ZIPF_7} \
#                    ${HRR_7} ${HR_MIN_7} ${HR_MAX_7} ${CR_MIN_7} ${CR_MAX_7}
# done


# # ==============================================================================
# # 图 8: 不同偏斜度下的鲁棒性 (Varying Zipf Alpha)
# # ==============================================================================

# # 固定参数
# TOPK_8=6
# # Skew Config for Fig 8
# HRR_8=0.25
# HR_MIN_8=0.2
# HR_MAX_8=0.3
# CR_MIN_8=0.1
# CR_MAX_8=0.2

# zipfs=(0.0 0.25 0.5 0.75 0.99)

# # 图 8(a): 8x8 集群, 1G 负载 (Batch=512)
# echo -e "\n \nStarting Figure 8(a) Experiments (8x8 Nodes, 1GB Load) " | tee -a ${LOG_FILE}
# BATCH_8A=512
# for z in "${zipfs[@]}"; do
#     run_experiment "Fig8a_Zipf_${z}" 8 ${BATCH_8A} ${TOPK_8} $z \
#                    ${HRR_8} ${HR_MIN_8} ${HR_MAX_8} ${CR_MIN_8} ${CR_MAX_8}
# done

# # 图 8(b): 16x8 集群, 1G 负载 (Batch=1024)
# echo -e "\n \nStarting Figure 8(b) Experiments (16x8 Nodes, 1GB Load) " | tee -a ${LOG_FILE}
# BATCH_8B=1024
# for z in "${zipfs[@]}"; do
#     run_experiment "Fig8b_Zipf_${z}" 16 ${BATCH_8B} ${TOPK_8} $z \
#                    ${HRR_8} ${HR_MIN_8} ${HR_MAX_8} ${CR_MIN_8} ${CR_MAX_8}
# done


# ==============================================================================
# 图 9: 路由稀疏度适应性 (Varying Top-K)
# ==============================================================================

# 固定参数
ZIPF_9=0.25
# 16x8 集群, 512MB 负载 -> Batch=512 (参考图7b)
BATCH_9=256
NODES_9=8
# Skew Config for Fig 9 (Same as Fig 8)
HRR_9=0.25
HR_MIN_9=0.2
HR_MAX_9=0.3
CR_MIN_9=0.1
CR_MAX_9=0.2

topks=(2 4 6 8)

echo -e "\n \nStarting Figure 9 Experiments (8x8 Nodes, 512MB Load)" | tee -a ${LOG_FILE}
for k in "${topks[@]}"; do
    run_experiment "Fig9_TopK_${k}" ${NODES_9} ${BATCH_9} $k ${ZIPF_9} \
                   ${HRR_9} ${HR_MIN_9} ${HR_MAX_9} ${CR_MIN_9} ${CR_MAX_9}
done

echo -e "\n All Experiments Completed Successfully!" | tee -a ${LOG_FILE}
echo "Log saved at: ${LOG_FILE}" | tee -a ${LOG_FILE}
