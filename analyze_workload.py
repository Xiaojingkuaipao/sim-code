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
    

if __name__ == "__main__":
    args = get_args()
    config = get_config(args)
    analyze_pkl_workload(args.workload_output_dir, config.total_gpus)