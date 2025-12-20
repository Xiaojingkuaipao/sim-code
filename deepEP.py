import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from common import *


class DeepEPSimulator:
    def __init__(self, config: SimConfig, cluster: Cluster):
        self.config = config
        self.cluster = cluster

        self.gpus = []
        for i in range(config.total_gpus):
            node_id = i // config.gpus_per_node
            local_rank = i % config.gpus_per_node
            self.gpus.append(GPU(i, node_id, local_rank))
    
    def get_gpu_by_node_rank(self, node_id, rank) -> GPU:
        '''
        通过node id和rank得到GPU对象
        '''
        global_id = node_id * self.config.gpus_per_node + rank
        return self.gpus[global_id]
    
    def run_simulation(self, workload_dir):
        '''
        给定通信需求，运行deepEP仿真,并输出仿真结果
        - 训练iter_num 的总延迟 s
        - 平均延迟 ms
        - 跨机流量 GB
        - 算法带宽 GB/s
        '''
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
        print(f"DeepEP Final Results")
        print(f"="*50)
        print(f"Total Latency            : {total_latency:.6f} s")
        print(f"Avg Layer Latency        : {avg_latency * 1000:.4f} ms")
        print(f"Total Inter-node Traffic : {total_inter_traffic / 1024 / 1024:.4f} GB")
        print(f"Algorithmic Bandwidth    : {algo_bw / 1024 / 1024:.4f} GB/s")
        print(f"="*50)

        
    
    def simulate_layer(self, request: list[TokenRequest]):
        '''
        给定一个layer的token request
        返回这一层的通信时间（s），以及这一层的通信量（kb）
        '''
        for gpu in self.gpus:
            gpu.reset()

        layer_inter_traffic_tokens = 0

        for req in request:
            # 处理每一个token请求
            src_gpu = self.gpus[req.src_gpu]

            # 因为会采用节点集合机制，所以在这里用节点分好组
            target_node_gpu = {} # {node_0:{gpu_0, gpu_1}}
            for target_gpu_id in req.target_gpus:
                target_node_id = target_gpu_id // self.config.gpus_per_node
                if target_node_id not in target_node_gpu:
                    target_node_gpu[target_node_id] = []
                target_node_gpu[target_node_id].append(target_gpu_id)
            
            # 对所有的目的GPU模拟deepEP分发
            for target_node_id, target_gpu_ids in target_node_gpu.items():
                
                # 如果是机间通信
                if src_gpu.node_id != target_node_id:
                    proxy_gpu = self.get_gpu_by_node_rank(target_node_id, src_gpu.local_rank)

                    proxy_gpu.inter_rx += 1
                    src_gpu.inter_tx += 1
                    layer_inter_traffic_tokens += 1

                    for target_gpu_id in target_gpu_ids:
                        if target_gpu_id != proxy_gpu.id:
                            proxy_gpu.intra_tx += 1
                            self.gpus[target_gpu_id].intra_rx += 1
                # 如果是机内通信
                else:
                    for target_gpu_id in target_gpu_ids:
                        if target_gpu_id != src_gpu.id:
                            self.gpus[target_gpu_id].intra_rx += 1
                            src_gpu.intra_tx += 1
        
        # 到这里一层的所有的token请求就都处理完成了
        # 因为机内转发和机间转发是同时进行的，所以all-to-all通信完成的时间取决于最慢的那个GPU的机内/机间通信操作
        max_inter_load = 0
        max_intra_load = 0

        for gpu in self.gpus:
            max_inter_load = max(max_inter_load, gpu.inter_rx, gpu.inter_tx)
            max_intra_load = max(max_intra_load, gpu.intra_rx, gpu.intra_tx)
        
        time_inter = (max_inter_load * self.config.token_size) / self.config.bw_inter
        time_intra = (max_intra_load * self.config.token_size) / self.config.bw_intra

        layer_time = max(time_inter, time_intra)

        return layer_time, layer_inter_traffic_tokens * self.config.token_size
    
if __name__ == "__main__":
    print("Test DeepEP Simulation")
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    sim = DeepEPSimulator(config, cluster)
    sim.run_simulation(args.workload_output_dir)