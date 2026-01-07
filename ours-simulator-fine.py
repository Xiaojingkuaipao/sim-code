import os
import pickle
import numpy as np
from common import *
from collections import defaultdict, deque
from analyze_workload import analyze_post_workload
import time


class OurSimulator:
    def __init__(self, config: SimConfig, cluster: Cluster):
        self.config = config
        self.cluster = cluster
        self.gpus :list[GPU] = []

        for i in range(config.total_gpus):
            node_id = i // config.gpus_per_node
            local_rank = i % config.gpus_per_node
            self.gpus.append(GPU(i, node_id, local_rank))
    
    def get_gpu_id(self, node_id, local_rank):
        return node_id * self.config.gpus_per_node + local_rank
    
    def run_simulation(self, workload_dir):
        if not os.path.exists(workload_dir):
            raise FileNotFoundError(f"{workload_dir} not found")
        
        print(f"Start Our Simulation")
        total_latency = 0
        total_inter_traffic = 0
        for iter_idx in range(self.config.iter_num):
            pkl_path = os.path.join(workload_dir, f"iter_{iter_idx}_requests.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"{pkl_path} not found")
            with open(pkl_path, 'rb') as f:
                layers_request = pickle.load(f)
            
            for layer_request in layers_request:
                layer_time, layer_inter_traffic = self.simulate_layer(layer_request)
                total_latency += layer_time
                total_inter_traffic += layer_inter_traffic
        
        avg_latency = total_latency / (self.config.iter_num * self.config.num_layers)

        # algorithm bandwidth computation
        total_payload = self.config.num_tokens * self.config.token_size * self.config.num_layers * self.config.iter_num * self.config.top_k
        algo_bw = total_payload / (total_latency * self.config.total_gpus)
        
        print(f"\n" + '=' * 50)
        print(f"Our Simulation Result")
        print(f"=" * 50)
        print(f"Total Latency(JCT)          :{total_latency:.6f} s")
        print(f"Avg Layer Latency           :{avg_latency * 1000:.4f} ms")
        print(f"Total Inter-node Traffic    :{total_inter_traffic / 1024 / 1024:.4f} GB")
        print(f"Algorithmic Bandwidth       :{algo_bw / 1024 / 1024:.4f} GB/s")
        print(f"="*50)



    def simulate_layer(self, requests: list[TokenRequest]):
        '''
        给定一个layer的token request
        返回这一层的通信时间（s），以及这一层的通信量（kb）

        step1 将TokenRequest对象转成Flow对象
        step2 执行算法第二步，发送端负载均衡
        step3 执行算法第三步，实现接收端负载均衡
        step4 计算总开销
        '''

        # 上一轮结束后清空负载
        for gpu in self.gpus:
            gpu.reset()
        start = time.perf_counter()
        # 存储所有的(token node) 流
        all_flows: list[Flow] = []
        for req in requests:
            src_node = req.src_gpu // self.config.gpus_per_node
            # 存储(node_id, target_gpus)对，因为一个token request会产生多个token node对
            target_node_gpu = defaultdict(list)
            for target_gpu in req.target_gpus:
                target_node = target_gpu // self.config.gpus_per_node
                if src_node != target_node:
                    target_node_gpu[target_node].append(target_gpu)
                else:
                    if req.src_gpu != target_gpu:
                        self.gpus[req.src_gpu].intra_tx += 1
                        self.gpus[target_gpu].intra_rx += 1
            for target_node, target_gpus in target_node_gpu.items():
                all_flows.append(Flow(req.token_id, req.src_gpu, target_node, target_gpus))
        
        # step2 发送端负载均衡
        gpu_send_queue :defaultdict[int, deque[Flow]]= defaultdict(deque)
        gpu_send_loads = np.zeros((self.config.num_nodes, self.config.gpus_per_node), dtype=int)

        # 统计每个GPU的发送负载
        for flow in all_flows:
            flow.sender = flow.src_gpu
            gpu_send_queue[flow.src_gpu].append(flow)

            src_node = flow.src_gpu // self.config.gpus_per_node
            src_rank = flow.src_gpu % self.config.gpus_per_node
            gpu_send_loads[src_node][src_rank] += 1
        
        for node_id in range(self.config.num_nodes):
            current_loads = gpu_send_loads[node_id]

            max_iter = len(all_flows) // self.config.num_nodes + 100
            for _ in range(max_iter):
                g_h = np.argmax(current_loads)
                g_c = np.argmin(current_loads)

                max_load = current_loads[g_h]
                min_load = current_loads[g_c]

                if max_load <= min_load + 1:
                    break

                hot_gpu_id = self.get_gpu_id(node_id, g_h)
                cold_gpu_id = self.get_gpu_id(node_id, g_c)

                # 选择一个flow从hot gpu转移到cold gpu
                # 由于flow size都是1，所以选哪个都一样
                move_flow = gpu_send_queue[hot_gpu_id].pop()
                # 转移到cold_gpu上
                move_flow.sender = cold_gpu_id
                
                gpu_send_queue[cold_gpu_id].append(move_flow)
                current_loads[g_h] -= 1
                current_loads[g_c] += 1
        
        # step3 接收端负载均衡
        gpu_recv_queue :defaultdict[int, deque[Flow]] = defaultdict(deque)
        gpu_recv_load = np.zeros((self.config.num_nodes, self.config.gpus_per_node), dtype=int)
        
        # 初始化接收队列和接收负载矩阵
        for flow in all_flows:
            target_node = flow.target_node
            # method1:默认接收者和deepEP相同，为同号卡
            sender_rank = flow.sender % self.config.gpus_per_node
            receiver = target_node * self.config.gpus_per_node + sender_rank
            # method2:默认接收者为flow.target_gpus中的第一个gpu
            # TODO

            flow.receiver = receiver
            receiver_rank = receiver % self.config.gpus_per_node
            gpu_recv_queue[receiver].append(flow)
            gpu_recv_load[target_node][receiver_rank] += 1
        
        for node_id in range(self.config.num_nodes):
            current_loads = gpu_recv_load[node_id]

            max_iter = len(all_flows) // self.config.num_nodes + 100
            for _ in range(max_iter):
                g_h = np.argmax(current_loads)
                g_c = np.argmin(current_loads)

                max_load = current_loads[g_h]
                min_load = current_loads[g_c]

                if max_load <= min_load + 1:
                    break

                hot_gpu_id = self.get_gpu_id(node_id, g_h)
                cold_gpu_id = self.get_gpu_id(node_id, g_c)

                move_flow = gpu_recv_queue[hot_gpu_id].pop()
                move_flow.receiver = cold_gpu_id

                gpu_recv_queue[cold_gpu_id].append(move_flow)
                current_loads[g_h] -= 1
                current_loads[g_c] += 1
        
        end = time.perf_counter()
        print(f"调度总开销：{(end - start) * 1000:.3f} ms")
        # step4 计算总开销
        
        total_inter_token = 0
        for flow in all_flows:
            # 计算机间负载
            self.gpus[flow.sender].inter_tx += 1
            self.gpus[flow.receiver].inter_rx += 1
            total_inter_token += 1

            # 计算path2和path4的机内中继负载
            if flow.src_gpu != flow.sender:
                self.gpus[flow.src_gpu].intra_tx += 1
                self.gpus[flow.sender].intra_rx += 1
            
            # 计算path3和path4的目的机内转发负载
            for target_gpu in set(flow.target_gpus):
                if flow.receiver != target_gpu:
                    self.gpus[flow.receiver].intra_tx += 1
                    self.gpus[target_gpu].intra_rx += 1
        
        # 到这里计算完负载了
        
        # 开始计算瓶颈
        max_inter_load = 0
        max_intra_load = 0

        for gpu in self.gpus:
            max_inter_load = max(max_inter_load, gpu.inter_rx, gpu.inter_tx)
            max_intra_load = max(max_intra_load, gpu.intra_rx, gpu.intra_tx)

        analyze_post_workload(self.gpus, "ours.png")
        
        time_inter = (max_inter_load * self.config.token_size) / self.config.bw_inter
        time_intra = (max_intra_load * self.config.token_size) / self.config.bw_intra

        layer_time = max(time_inter, time_intra)
        
        return layer_time, total_inter_token * self.config.token_size

if __name__ == "__main__":
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    simulator = OurSimulator(config, cluster)
    simulator.run_simulation(args.workload_output_dir)