import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from common import *

def analyze_pkl_workload(workload_dir, total_gpus=32):
    """
    统计 workload 文件夹下的所有 pkl 文件，绘制每个 GPU 的收发负载
    """
    if not os.path.exists(workload_dir):
        print(f"Error: Directory '{workload_dir}' not found.")
        return

    # recv_counts[i] 表示 gpu i 接收了多少个 Token
    recv_counts = np.zeros(total_gpus, dtype=int)

    # 获取所有 pkl 文件
    pkl_files = glob.glob(os.path.join(workload_dir, "*.pkl"))
    if not pkl_files:
        print("No .pkl files found in the directory.")
        return

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            layers_requests = pickle.load(f)
            
            for layer in layers_requests:
                for req in layer:                    
                    # 统计接收端
                    for tgt_gpu in req.target_gpus:
                        recv_counts[tgt_gpu] += 1

    print("\n" + "="*40)
    print("Workload Summary")
    print("="*40)
    print(f"Total Recv Tokens     : {np.sum(recv_counts)}")
    print("-" * 40)
    print(f"Max Recv by single GPU: {np.max(recv_counts)} (GPU {np.argmax(recv_counts)})")
    print(f"Min Recv by single GPU: {np.min(recv_counts)} (GPU {np.argmin(recv_counts)})")
    print("="*40)

    plt.figure(figsize=(15, 6))

    plt.bar(range(total_gpus), recv_counts, color='salmon', edgecolor='red')
    plt.title('Token Receive Count per GPU')
    plt.xlabel('GPU ID')
    plt.ylabel('Number of Tokens Received')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = "workload_distribution.png"
    plt.savefig(save_path)

def analyze_post_workload(gpus :list[GPU], save_path=None):
    '''
    接收simulator最后的gpu list，绘制调度完成之后的gpu通信负载柱状图，并保存在save_path中
    '''

    inter_send_count = np.zeros(len(gpus), dtype=int)
    inter_recv_count = np.zeros(len(gpus), dtype=int)
    intra_send_count = np.zeros(len(gpus), dtype=int)
    intra_recv_count = np.zeros(len(gpus), dtype=int)

    for gpu in gpus:
        inter_send_count[gpu.id] += gpu.inter_tx
        inter_recv_count[gpu.id] += gpu.inter_rx
        intra_send_count[gpu.id] += gpu.intra_tx
        intra_recv_count[gpu.id] += gpu.intra_rx
    
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.bar(range(len(gpus)), inter_send_count)
    plt.title("Inter-node Token Send Count per GPU")
    plt.xlabel("GPU ID")
    plt.ylabel("Number of Tokens Send")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 2)
    plt.bar(range(len(gpus)), inter_recv_count)
    plt.title("Inter-node Token Recv Count per GPU")
    plt.xlabel("GPU ID")
    plt.ylabel("Number of Tokens Recv")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 3)
    plt.bar(range(len(gpus)), intra_send_count)
    plt.title("Intra-node Token Send Count per GPU")
    plt.xlabel("GPU ID")
    plt.ylabel("Number of Tokens Send")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 4)
    plt.bar(range(len(gpus)), intra_recv_count)
    plt.title("Intra-node Token Recv Count per GPU")
    plt.xlabel("GPU ID")
    plt.ylabel("Number of Tokens Send")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path is not None:
        plt.savefig(save_path)
    
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
    args = get_args()
    config = get_config(args)
    analyze_pkl_workload(args.workload_output_dir, config.total_gpus)