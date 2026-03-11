import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from common import *
import matplotlib.pyplot as plt

def generate_workload_bimodal_skew(
        config: SimConfig, 
        cluster: Cluster, 
        output_dir: str,  # 负载.pkl输出到的文件夹
        hot_rank_ratio: float = 0.5,  
        hot_remote_ratio: list[float] = [0.4, 0.5], 
        cold_remote_ratio: list[float] = [0.1, 0.2]
):
    """
    生成同时具备 [源端Rank偏斜] 和 [目标Zipf偏斜] 的负载。
    使用 hot_rank_ratio 动态选择 hot ranks，并使用区间随机选择 remote ratio。
    """
    # --- 1. 参数验证 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 验证 hot_remote_ratio 和 cold_remote_ratio 是否为长度为 2 的列表且在 [0, 1] 范围内
    if len(hot_remote_ratio) != 2 or not all(0 <= r <= 1.0 for r in hot_remote_ratio):
        raise ValueError(f"hot_remote_ratio must be a list of 2 floats in [0, 1], got {hot_remote_ratio}")
    if len(cold_remote_ratio) != 2 or not all(0 <= r <= 1.0 for r in cold_remote_ratio):
        raise ValueError(f"cold_remote_ratio must be a list of 2 floats in [0, 1], got {cold_remote_ratio}")
    if not (0 <= hot_rank_ratio <= 1.0):
         raise ValueError(f"hot_rank_ratio must be in [0, 1], got {hot_rank_ratio}")

    # 验证专家数量是否足够
    if config.total_experts < config.top_k:
        raise ValueError(f"Total experts ({config.total_experts}) must be >= top_k ({config.top_k})")

    # 确定每个 Node 的 Hot Ranks
    num_hot_ranks = int(config.gpus_per_node * hot_rank_ratio)
    # 后续为每个节点独立随机选择 hot ranks
    
    print(f"Start Generate Bimodal Skew Workloads...")
    print(f" - Hot Rank Ratio: {hot_rank_ratio} (Num Hot Ranks: {num_hot_ranks}/{config.gpus_per_node})")
    print(f" - Hot Remote Ratio Range: {hot_remote_ratio}")
    print(f" - Cold Remote Ratio Range: {cold_remote_ratio}")
    print(f" - Global Target Zipf Alpha: {config.zipf_alpha}")

    # 每个 GPU 的固定 Token 数量 (严格整除，丢弃余数以保证均匀)
    if config.num_tokens % config.total_gpus != 0:
        print(f"Warning: num_tokens ({config.num_tokens}) is not divisible by total_gpus ({config.total_gpus}). Remainder will be dropped.")
    
    tokens_per_gpu = config.num_tokens // config.total_gpus
    
    # 预先计算全局专家的 Zipf 概率分布 (用于跨机 Token 选择专家)
    ranks = np.arange(1, config.total_experts + 1) 
    weights = 1.0 / np.power(ranks, config.zipf_alpha) # 每个专家的权重 1 / (k^{\alpha})
    zipf_prob_vector = weights / np.sum(weights) # p(k) = (1 / k) / (\sum_{n=1}^{N}{1 / n^{\alpha}})

    total_generated_count = 0
    
    # 存储每个 GPU 的实际配置以便后续分析 (保存到元数据文件中)
    # gpu_config_metadata[gpu_id] = {'is_hot': bool, 'remote_ratio': float}
    gpu_config_metadata = {}

    for iter_idx in range(config.iter_num): 
        iter_data = []
        print(f"Generating Iter:{iter_idx}/{config.iter_num - 1}...")
        
        # 重置每轮迭代的 token_id 计数器
        current_iter_token_id = 0
        
        # 为了简化，我们在第一次进入循环时生成配置，并保存到 metadata
        if iter_idx == 0:
            for node_id in range(config.num_nodes):
                # 每个节点独立随机选择 hot ranks
                node_ranks = np.random.choice(config.gpus_per_node, num_hot_ranks, replace=False)
                node_hot_ranks_set = set(node_ranks)
                
                for local_rank in range(config.gpus_per_node):
                    gpu_id = node_id * config.gpus_per_node + local_rank
                    is_hot = local_rank in node_hot_ranks_set
                    
                    # 随机选择 remote ratio
                    if is_hot:
                        ratio = np.random.uniform(hot_remote_ratio[0], hot_remote_ratio[1])
                    else:
                        ratio = np.random.uniform(cold_remote_ratio[0], cold_remote_ratio[1])
                    
                    gpu_config_metadata[gpu_id] = {
                        'is_hot': is_hot,
                        'remote_ratio': ratio
                    }
        
        # 保存 metadata 到输出目录 (仅一次)
        if iter_idx == 0:
            with open(os.path.join(output_dir, "workload_metadata.pkl"), "wb") as f:
                pickle.dump(gpu_config_metadata, f)

        for layer_id in tqdm(range(config.num_layers), desc="Layers"): 
            layer_request = []
            
            # 遍历每一个物理 GPU
            for gpu_id in range(config.total_gpus):
                node_id = gpu_id // config.gpus_per_node
                # local_rank = gpu_id % config.gpus_per_node # 不再直接需要，因为已经存入 metadata
                
                # 获取该 GPU 的配置
                gpu_conf = gpu_config_metadata[gpu_id]
                current_remote_ratio = gpu_conf['remote_ratio']
                
                # 划分本地专家和远程专家
                local_expert_ids = [e for e, g in cluster.expert_to_gpu.items() if g // config.gpus_per_node == node_id]
                local_expert_set = set(local_expert_ids)
                remote_expert_ids_count = config.total_experts - len(local_expert_ids)

                # 检查专家数量是否满足 top_k
                if len(local_expert_ids) < config.top_k:
                     raise ValueError(f"Node {node_id} has {len(local_expert_ids)} experts, but top_k is {config.top_k}. Cannot satisfy local request without replacement.")
                
                if remote_expert_ids_count < config.top_k:
                    raise ValueError(f"Node {node_id} has {remote_expert_ids_count} remote experts, but top_k is {config.top_k}. Cannot satisfy remote request.")

                # 计算跨机和本地 Token 数量
                num_remote = int(tokens_per_gpu * current_remote_ratio)
                num_local = tokens_per_gpu - num_remote

                # 1. 生成 Local Tokens
                for _ in range(num_local):
                    candidates = np.random.choice(local_expert_ids, config.top_k, replace=False) 
                    target_experts = [int(exp) for exp in candidates]
                        
                    req = TokenRequest(token_id=current_iter_token_id, src_gpu=gpu_id, target_experts=target_experts)
                    cluster.resolve_targets(req)
                    layer_request.append(req)
                    current_iter_token_id += 1
                    total_generated_count += 1

                # 2. 生成 Remote Tokens (带 Zipf 偏斜)
                for _ in range(num_remote):
                    target_experts = []
                    
                    while len(target_experts) < config.top_k:
                        candidates = np.random.choice(
                            config.total_experts,
                            size=config.top_k * 2,
                            p=zipf_prob_vector,
                            replace=True
                        )
                        
                        for exp in candidates:
                            exp = int(exp)
                            if exp not in local_expert_set and exp not in target_experts:
                                target_experts.append(exp)
                                if len(target_experts) == config.top_k:
                                    break
                    
                    req = TokenRequest(token_id=current_iter_token_id, src_gpu=gpu_id, target_experts=target_experts)
                    cluster.resolve_targets(req)
                    layer_request.append(req)
                    current_iter_token_id += 1
                    total_generated_count += 1
                    
            iter_data.append(layer_request)
        
        # 保存数据
        file_path = os.path.join(output_dir, f"iter_{iter_idx}_requests.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(iter_data, f)
        
        del iter_data

    print(f"Workload Generation Complete!")
    print(f"Generated {total_generated_count} token requests in {output_dir}")


def analyze_bimodal_skew(workload_dir, config, cluster):
    
    print("\nStarting Analysis of Bimodal Skew Workload...")
    
    # 尝试加载元数据
    metadata_path = os.path.join(workload_dir, "workload_metadata.pkl")
    gpu_metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            gpu_metadata = pickle.load(f)
        print("Loaded workload metadata for verification.")
    else:
        print("Warning: No workload metadata found. Verification might be inaccurate for dynamic workloads.")

    # 1. Load Data
    pkl_files = [f for f in os.listdir(workload_dir) if f.endswith(".pkl") and f.startswith("iter_")]
    all_requests = []
    for pkl_file in pkl_files:
        with open(os.path.join(workload_dir, pkl_file), 'rb') as f:
            iter_data = pickle.load(f)
            for layer in iter_data:
                all_requests.extend(layer)
    
    print(f"Loaded {len(all_requests)} total requests.")

    # 2. Analyze Source Skew (Local vs Remote)
    src_stats = {gpu_id: {'local': 0, 'remote': 0} for gpu_id in range(config.total_gpus)}
    
    # 3. Analyze Target Skew (Zipf)
    remote_target_expert_counts = np.zeros(config.total_experts, dtype=int)
    
    for req in all_requests:
        src_node = req.src_gpu // config.gpus_per_node
        for exp_id in req.target_experts:
            target_gpu = cluster.expert_to_gpu[exp_id]
            target_node = target_gpu // config.gpus_per_node
            
            if src_node == target_node:
                src_stats[req.src_gpu]['local'] += 1
            else:
                src_stats[req.src_gpu]['remote'] += 1
                remote_target_expert_counts[exp_id] += 1

    # Plot 1: Source Skew
    plt.figure(figsize=(15, 6))
    gpu_ids = np.arange(config.total_gpus)
    local_counts = [src_stats[i]['local'] for i in gpu_ids]
    remote_counts = [src_stats[i]['remote'] for i in gpu_ids]
    total_counts = np.array(local_counts) + np.array(remote_counts)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        remote_ratios = np.array(remote_counts) / total_counts
        remote_ratios = np.nan_to_num(remote_ratios)

    plt.bar(gpu_ids, local_counts, label='Local Tokens', color='skyblue')
    plt.bar(gpu_ids, remote_counts, bottom=local_counts, label='Remote Tokens', color='salmon')
    
    # Mark hot ranks based on metadata if available
    for gpu_id in gpu_ids:
        is_hot = False
        if gpu_metadata:
            is_hot = gpu_metadata[gpu_id]['is_hot']
        else:
            raise FileNotFoundError("No metadata file found. Please generate workload with metadata.")
        
        if is_hot:
            plt.text(gpu_id, total_counts[gpu_id] + 5, 'HOT', ha='center', fontsize=8, color='red', fontweight='bold')
            
    plt.title('Source Skew: Local vs Remote Token Distribution per GPU')
    plt.xlabel('Source GPU ID')
    plt.ylabel('Number of Tokens')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(workload_dir, "analysis_source_skew.png"))
    print(f"Saved source skew plot to {os.path.join(workload_dir, 'analysis_source_skew.png')}")

    # Verify Ratios
    print("\n--- Verifying Source Skew Ratios ---")
    for gpu_id in range(config.total_gpus):
        actual = remote_ratios[gpu_id]
        expected_range = None
        
        if gpu_metadata:
            expected_val = gpu_metadata[gpu_id]['remote_ratio']
            # Since we use int() truncation in generation, actual ratio might be slightly lower
            status = "PASS" if abs(actual - expected_val) < 0.05 else "FAIL"
            print(f"GPU {gpu_id:02d}: Remote Ratio = {actual:.4f} (Target {expected_val:.4f}) -> {status}")
        else:
            # Fallback verification (approximate)
            # This part might be inaccurate if random ranges are used but metadata is missing
            print(f"GPU {gpu_id:02d}: Remote Ratio = {actual:.4f} (Metadata missing, cannot verify exact target)")

    # Plot 2: Target Skew
    plt.figure(figsize=(15, 6))
    sorted_indices = np.argsort(-remote_target_expert_counts)
    sorted_counts = remote_target_expert_counts[sorted_indices]
    ranks = np.arange(1, len(sorted_counts) + 1)
    
    plt.loglog(ranks, sorted_counts, 'o-', markersize=4, label='Actual Remote Traffic')
    if len(sorted_counts) > 0 and sorted_counts[0] > 0:
        theoretical_zipf = sorted_counts[0] * (ranks ** -config.zipf_alpha)
        plt.loglog(ranks, theoretical_zipf, 'r--', label=f'Theoretical Zipf (alpha={config.zipf_alpha})')

    plt.title('Target Skew: Remote Expert Popularity (Log-Log Scale)')
    plt.xlabel('Expert Rank')
    plt.ylabel('Access Frequency')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(workload_dir, "analysis_target_zipf.png"))
    print(f"Saved target zipf plot to {os.path.join(workload_dir, 'analysis_target_zipf.png')}")

if __name__ == "__main__":
    print("Test Generate Workload")
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    
    generate_workload_bimodal_skew(
        config=config,
        cluster=cluster,
        output_dir=args.workload_output_dir,
        hot_rank_ratio=args.hot_rank_ratio,
        hot_remote_ratio=args.hot_remote_ratio,
        cold_remote_ratio=args.cold_remote_ratio
    )
    analyze_bimodal_skew(args.workload_output_dir, config, cluster)