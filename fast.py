import os
import pickle
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import numpy as np
from common import *

class FASTSimulator:
    def __init__(self, config: SimConfig, cluster: Cluster):
        self.config = config
        self.cluster = cluster
    
    def run_simulation(self, workload_dir):
        if not os.path.exists(workload_dir):
            raise FileNotFoundError(f"{workload_dir} not found")
        
        total_latency = 0
        total_inter_traffic = 0
        for iter_idx in range(self.config.iter_num):
            pkl_path = os.path.join(workload_dir, f"iter_{iter_idx}_requests.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"{pkl_path} not found, pleas generate the workload first")
            with open(pkl_path, "rb") as f:
                layers_request = pickle.load(f)
            
            for layer_request in layers_request:
                layer_time, layer_inter_traffic = self.simulate_layer(layer_request)
                total_latency += layer_time
                total_inter_traffic += layer_inter_traffic
        
        # 结果统计
        avg_latency = total_latency / (self.config.iter_num * self.config.num_layers)
        
        # algorithmic bandwidth = (total transfer size) / (total latency * num of gpus)
        total_payload = self.config.num_tokens * self.config.token_size * self.config.num_layers * self.config.iter_num * self.config.top_k
        algo_bw = total_payload / (total_latency * self.config.total_gpus)

        print(f"="*50)
        print(f"FAST Final Results")
        print(f"="*50)
        print(f"Total Latency            : {total_latency:.6f} s")
        print(f"Avg Layer Latency        : {avg_latency * 1000:.4f} ms")
        print(f"Total Inter-node Traffic : {total_inter_traffic / 1024 / 1024:.4f} GB")
        print(f"Algorithmic Bandwidth    : {algo_bw / 1024 / 1024:.4f} GB/s")
        print(f"="*50)

    def simulate_layer(self, request: list[TokenRequest]):
        '''
        FAST 一层的仿真逻辑
        返回
        '''
        total_inter_traffic = 0
        total_latency = 0
        num_nodes = self.config.num_nodes
        # 服务器级别的通信矩阵
        node_traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        gpu_traffic_matrix = np.zeros((self.config.total_gpus, self.config.total_gpus), dtype=int)
        gpu_raw_send = defaultdict(int) # 用于记录最开始的每个GPU发送量，便于计算balancing的时间

        for req in request:
            gpu_raw_send[req.src_gpu] += len(req.target_gpus)

            for target_gpu in req.target_gpus:
                target_gpu_node = target_gpu // self.config.gpus_per_node
                src_gpu_node = req.src_gpu // self.config.gpus_per_node
                if src_gpu_node != target_gpu_node:
                    gpu_traffic_matrix[req.src_gpu][target_gpu] += 1
                    total_inter_traffic += 1

        gpus_per_node = self.config.gpus_per_node
        num_nodes = self.config.num_nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                row_start = i * gpus_per_node
                row_end = (i + 1) * gpus_per_node
                col_start = j * gpus_per_node
                col_end = (j + 1) * gpus_per_node
                sub_matrix = gpu_traffic_matrix[row_start:row_end, col_start:col_end]
                sub_matrix_sum = np.sum(sub_matrix)
                node_traffic_matrix[i][j] = sub_matrix_sum // gpus_per_node
        
        # 计算balance的时间，也是组成总时间的第一个阶段
        max_balance_transfer = 0
        for node_id in range(num_nodes):
            traffic_per_gpu = np.sum(node_traffic_matrix[node_id])
            start_gpu = node_id * gpus_per_node
            for i in range(gpus_per_node):
                transfer = abs(gpu_raw_send[start_gpu + i] - traffic_per_gpu)
                max_balance_transfer = max(transfer, max_balance_transfer)
        
        balance_latency = (max_balance_transfer * self.config.token_size) / self.config.bw_intra

        # padding node矩阵，用于brikhoff分解
        row_sums = np.sum(node_traffic_matrix, axis=1)
        col_sums = np.sum(node_traffic_matrix, axis=0)
        max_load = np.max([np.max(row_sums), np.max(col_sums)])

        # FAST已经论证了瓶颈仍然是瓶颈，跨机传输时间仍然是由于负载最大的那个node决定，所以可以靠这个式子计算出第二阶段总时间
        inter_traffic_latency = (max_load * self.config.token_size) / self.config.bw_inter

        padded_matrix, diff = self._pad_matrix(node_traffic_matrix, max_load)
        stages = self._brikhoff_decomposition(padded_matrix)
        last_stage, remain_padding = self._find_last_stage(stages, diff)
        last_stage = np.maximum(0, last_stage - remain_padding)
        max_val = np.max(last_stage)

        redist_tokens = max_val / 2
        redist_latency = (redist_tokens * self.config.token_size) / self.config.bw_intra

        total_latency += balance_latency + inter_traffic_latency + redist_latency

        return total_latency, total_inter_traffic * self.config.token_size

    
    def _brikhoff_decomposition(self, matrix):
        """
        给定一个双随机矩阵，进行brikhoff分解并返回stages
        """
        stages = []
        residual = matrix.copy().astype(int)
        N = matrix.shape[0]

        while np.any(residual > 0):
            cost_matrix = np.where(residual > 0, 0, 10**9)

            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            vals = residual[row_idx, col_idx]
            alpha = int(np.min(vals))

            if alpha == 0: break

            stage_matrix = np.zeros((N, N), dtype=int)
            stage_matrix[row_idx, col_idx] = alpha
            stages.append(stage_matrix)
            residual[row_idx, col_idx] -= alpha
        
        return stages
    
    def _pad_matrix(self, matrix, target_sum):
        """
        给定一个矩阵和目标和，将它填充为双随机矩阵
        返回这个双随机矩阵，并返回填充矩阵
        """
        N = matrix.shape[0]
        padded = matrix.copy()
        # 用于记录在哪些地方填了虚拟流量
        diff = np.zeros_like(padded, dtype=int)
        
        row_sum = np.sum(matrix, axis=1)
        col_sum = np.sum(matrix, axis=0)

        r = 0
        c = 0
        while r < N and c < N:
            row_slack = target_sum - row_sum[r]
            col_slack = target_sum - col_sum[c]

            if row_slack == 0:
                r += 1
                continue
            if col_slack == 0:
                c += 1
                continue

            fill_val = min(row_slack, col_slack)

            padded[r, c] += fill_val

            diff[r, c] += fill_val

            row_sum[r] += fill_val
            col_sum[c] += fill_val
            if row_sum[r] == target_sum:
                r += 1
            if col_sum[c] == target_sum:
                c += 1

        return padded, diff
    
    def _find_last_stage(self, stages, diff):
        """
        找到最后一个包含真实流量（非 Padding）的阶段索引或阶段本身。
        """
        remaining_padding = diff.copy().astype(int)
        
        # 从后往前遍历索引
        for i in range(len(stages) - 1, -1, -1):
            stage = stages[i]
            
            # 检查当前 stage 是否完全由填充组成
            if np.any(stage > remaining_padding):
                return stage, remaining_padding
            
            remaining_padding -= stage
            
        return stages[0], remaining_padding
    
if __name__ == "__main__":
    print("Test FAST Simulation")
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    sim = FASTSimulator(config, cluster)
    sim.run_simulation(args.workload_output_dir)