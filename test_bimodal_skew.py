
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from generate_workload import generate_workload_bimodal_skew
from common import SimConfig, Cluster, TokenRequest

def analyze_bimodal_skew(workload_dir, config, hot_ranks, hot_remote_ratio, cold_remote_ratio):
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

    # --- Plot 1: Source Skew (Local vs Remote Ratio) ---
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

    # --- Plot 2: Target Skew (Remote Token Destination Zipf) ---
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
    # 1. Setup Configuration
    # Using small scale for quick testing, but large enough to show patterns
    config = SimConfig(
        num_nodes=4,
        gpus_per_node=8,
        experts_per_gpu=2,  # Total 64 experts
        num_layers=2,
        iter_num=2,
        seq_len=1024,       # Smaller seq_len for faster generation
        batch_size=4,
        top_k=2,
        zipf_alpha=1.2
    )
    
    cluster = Cluster(config)
    output_dir = "./test_workload_bimodal"
    
    hot_ranks = [0, 1]
    hot_remote_ratio = 0.8
    cold_remote_ratio = 0.2
    
    # 2. Generate Workload
    print("Generating Workload...")
    generate_workload_bimodal_skew(
        config=config,
        cluster=cluster,
        output_dir=output_dir,
        hot_ranks=hot_ranks,
        hot_remote_ratio=hot_remote_ratio,
        cold_remote_ratio=cold_remote_ratio
    )
    
    # 3. Analyze Workload
    analyze_bimodal_skew(output_dir, config, hot_ranks, hot_remote_ratio, cold_remote_ratio)
