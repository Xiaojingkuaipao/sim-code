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
    """
    # --- 1. 参数验证 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not (0 <= hot_remote_ratio <= 1.0):
        raise ValueError(f"hot_remote_ratio must be in [0, 1], got {hot_remote_ratio}")
    if not (0 <= cold_remote_ratio <= 1.0):
        raise ValueError(f"cold_remote_ratio must be in [0, 1], got {cold_remote_ratio}")
    
    # 验证 hot_ranks 是否在合法范围内
    if any(r < 0 or r >= config.gpus_per_node for r in hot_ranks):
        raise ValueError(f"hot_ranks elements must be in [0, {config.gpus_per_node - 1}]")

    # 验证专家数量是否足够
    # 注意：这里做的是全局检查，更严格的检查在循环内部针对每个节点进行
    if config.total_experts < config.top_k:
        raise ValueError(f"Total experts ({config.total_experts}) must be >= top_k ({config.top_k})")

    print(f"Start Generate Bimodal Skew Workloads...")
    print(f" - Hot Ranks: {hot_ranks} (Remote Ratio: {hot_remote_ratio*100}%)")
    print(f" - Cold Ranks Remote Ratio: {cold_remote_ratio*100}%")
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

    for iter_idx in range(config.iter_num): # 这个循环相当于没有，因为多层的结果都是一样的，所以默认的config.iter_num=1
        iter_data = []
        print(f"Generating Iter:{iter_idx}/{config.iter_num - 1}...")
        
        # 重置每轮迭代的 token_id 计数器
        current_iter_token_id = 0

        # 这个循环也相当于没有，因为不管多少层输入负载的分布一致，结果差不太多，所以只需要一层即可
        for layer_id in tqdm(range(config.num_layers), desc="Layers"): 
            layer_request = []
            
            # 遍历每一个物理 GPU
            for gpu_id in range(config.total_gpus):
                node_id = gpu_id // config.gpus_per_node
                local_rank = gpu_id % config.gpus_per_node
                
                # 划分本地专家(在同一个node内)和远程专家(在其他node中)
                local_expert_ids = [e for e, g in cluster.expert_to_gpu.items() if g // config.gpus_per_node == node_id]
                local_expert_set = set(local_expert_ids)
                remote_expert_ids_count = config.total_experts - len(local_expert_ids) # 远程专家的数量

                # 检查专家数量是否满足 top_k
                if len(local_expert_ids) < config.top_k:
                     # 如果本地专家不足，无法生成纯本地的不重复 Token
                     # 这里选择抛出异常，或者根据需求改为允许重复
                     raise ValueError(f"Node {node_id} has {len(local_expert_ids)} experts, but top_k is {config.top_k}. Cannot satisfy local request without replacement.")
                
                if remote_expert_ids_count < config.top_k:
                    raise ValueError(f"Node {node_id} has {remote_expert_ids_count} remote experts, but top_k is {config.top_k}. Cannot satisfy remote request.")

                # 根据 Rank 决定跨机比例
                if local_rank in hot_ranks: # 如果当前rank是hot rank(机间发送负载重的rank)
                    num_remote = int(tokens_per_gpu * hot_remote_ratio) # 当前GPU需要跨机发送的token数
                else:
                    num_remote = int(tokens_per_gpu * cold_remote_ratio)
                
                num_local = tokens_per_gpu - num_remote # 目的地在本机内的token数量

                # 1. 生成 Local Tokens
                for _ in range(num_local):
                    # 本地 Token 随机选择本地专家 (replace=False表示无放回抽样，一个token不会选择到同一个专家两次)
                    candidates = np.random.choice(local_expert_ids, config.top_k, replace=False) 
                    target_experts = [int(exp) for exp in candidates] # 本地专家的id列表
                        
                    req = TokenRequest(token_id=current_iter_token_id, src_gpu=gpu_id, target_experts=target_experts)
                    cluster.resolve_targets(req)
                    layer_request.append(req)
                    current_iter_token_id += 1
                    total_generated_count += 1

                # 2. 生成 Remote Tokens (带 Zipf 偏斜)
                for _ in range(num_remote):
                    target_experts = []
                    
                    # 使用 while 循环配合批量采样
                    while len(target_experts) < config.top_k:
                        # 批量采样以提高效率 (2倍 top_k 通常足够)
                        candidates = np.random.choice(
                            config.total_experts,
                            size=config.top_k * 2,
                            p=zipf_prob_vector,
                            replace=True
                        )
                        
                        for exp in candidates:
                            exp = int(exp)
                            # 必须是远程专家 (不在 local_expert_set 中) 且未被选过
                            if exp not in local_expert_set and exp not in target_experts:
                                target_experts.append(exp)
                                if len(target_experts) == config.top_k:
                                    break
                    
                    req = TokenRequest(token_id=current_iter_token_id, src_gpu=gpu_id, target_experts=target_experts)
                    cluster.resolve_targets(req)
                    layer_request.append(req)
                    current_iter_token_id += 1
                    total_generated_count += 1
                    
            # TODO 这个token可能既被发送到本地GPU上,又发到远端GPU上,上述做法有些绝对,后续商榷是否有更好的方法
            iter_data.append(layer_request)
        
        # 保存数据
        file_path = os.path.join(output_dir, f"iter_{iter_idx}_requests.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(iter_data, f)
        
        del iter_data

    print(f"Workload Generation Complete!")
    print(f"Generated {total_generated_count} token requests in {output_dir}")


def analyze_bimodal_skew(workload_dir, config, hot_ranks, hot_remote_ratio, cold_remote_ratio, cluster):
    print("\nStarting Analysis of Bimodal Skew Workload...")
    
    # 1. Load Data
    pkl_files = [f for f in os.listdir(workload_dir) if f.endswith(".pkl")]
    all_requests = []
    for pkl_file in pkl_files:
        with open(os.path.join(workload_dir, pkl_file), 'rb') as f:
            iter_data = pickle.load(f)
            # iter_data is a list of layers, each layer is a list of requests
            for layer in iter_data:
                all_requests.extend(layer)
    
    print(f"Loaded {len(all_requests)} total requests.")

    # 2. Analyze Source Skew (Local vs Remote)
    # count[gpu_id] = {'local': 0, 'remote': 0}
    src_stats = {gpu_id: {'local': 0, 'remote': 0} for gpu_id in range(config.total_gpus)}
    
    # 3. Analyze Target Skew (Zipf) - Only for Remote Tokens
    remote_target_expert_counts = np.zeros(config.total_experts, dtype=int)
    
    for req in all_requests:
        src_node = req.src_gpu // config.gpus_per_node
        
        # Check each target expert
        for exp_id in req.target_experts:
            target_gpu = cluster.expert_to_gpu[exp_id]
            target_node = target_gpu // config.gpus_per_node
            
            if src_node == target_node:
                src_stats[req.src_gpu]['local'] += 1
            else:
                src_stats[req.src_gpu]['remote'] += 1
                remote_target_expert_counts[exp_id] += 1

    # Plot 1: Source Skew (Local vs Remote Ratio)
    plt.figure(figsize=(15, 6))
    gpu_ids = np.arange(config.total_gpus)
    local_counts = [src_stats[i]['local'] for i in gpu_ids]
    remote_counts = [src_stats[i]['remote'] for i in gpu_ids]
    total_counts = np.array(local_counts) + np.array(remote_counts)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        remote_ratios = np.array(remote_counts) / total_counts
        remote_ratios = np.nan_to_num(remote_ratios)

    plt.bar(gpu_ids, local_counts, label='Local Tokens', color='skyblue')
    plt.bar(gpu_ids, remote_counts, bottom=local_counts, label='Remote Tokens', color='salmon')
    
    # Mark hot ranks
    for gpu_id in gpu_ids:
        if gpu_id % config.gpus_per_node in hot_ranks:
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
        local_rank = gpu_id % config.gpus_per_node
        expected = hot_remote_ratio if local_rank in hot_ranks else cold_remote_ratio
        actual = remote_ratios[gpu_id]
        status = "PASS" if abs(actual - expected) < 0.05 else "FAIL" # Allow some deviation due to integer rounding
        print(f"GPU {gpu_id:02d} (Rank {local_rank}): Remote Ratio = {actual:.4f} (Expected {expected:.2f}) -> {status}")

    # Plot 2: Target Skew (Remote Token Destination Zipf) 
    plt.figure(figsize=(15, 6))
    
    # Sort experts by frequency to see the curve
    sorted_indices = np.argsort(-remote_target_expert_counts)
    sorted_counts = remote_target_expert_counts[sorted_indices]
    ranks = np.arange(1, len(sorted_counts) + 1)
    
    plt.loglog(ranks, sorted_counts, 'o-', markersize=4, label='Actual Remote Traffic')
    
    # Theoretical Zipf line (fitted to the first point)
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
    analyze_bimodal_skew(
        args.workload_output_dir, config, hot_ranks, hot_remote_ratio, cold_remote_ratio, cluster
    )