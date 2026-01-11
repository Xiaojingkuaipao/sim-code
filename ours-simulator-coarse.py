from common import *
from analyze_workload import *
import pickle
import numpy as np
import os
from collections import defaultdict

class Flow:
    def __init__(self, flow_id, src_gpu, target_node, size):
        self.id = flow_id
        self.src_gpu = src_gpu
        self.target_node = target_node
        self.size = size

        self.sender = src_gpu
        self.receiver = -1

        self.requests :list[TokenRequest] = []
        
class OurSimulatorCoarse:
    def __init__(self, config :SimConfig, cluster :Cluster):
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
        
        total_latency = 0
        total_inter_traffic = 0
        
        for iter_idx in range(self.config.iter_num):
            pkl_path = os.path.join(workload_dir, f"iter_{iter_idx}_requests.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"{pkl_path} not found.")
            
            with open(pkl_path, "rb") as f:
                layers_request = pickle.load(f)
            
            for layer_request in layers_request:
                layer_time, layer_inter_traffic = self.simulate_layer(layer_request)
                total_latency += layer_time
                total_inter_traffic += layer_inter_traffic
        
        # 结果统计
        avg_latency = total_latency / (self.config.iter_num * self.config.num_layers)
        
        # Algorithmic Bandwidth
        total_payload = self.config.num_tokens * self.config.token_size * self.config.num_layers * self.config.iter_num * self.config.top_k
        algo_bw = total_payload / (total_latency * self.config.total_gpus)

        print(f"\n" + "="*50)
        print(f"Ours (Coarse) Final Results")
        print(f"="*50)
        print(f"Total Latency (JCT)      : {total_latency:.6f} s")
        print(f"Avg Layer Latency        : {avg_latency * 1000:.4f} ms")
        print(f"Total Inter-node Traffic : {total_inter_traffic / 1024 / 1024:.4f} GB")
        print(f"Algorithmic Bandwidth    : {algo_bw / 1024 / 1024:.4f} GB/s")
        print(f"="*50)
    
    def simulate_layer(self, requests :list[TokenRequest]):
        '''
        给定一个layer的token request
        返回这一层的通信时间（s），以及这一层的通信量（kb）

        step1 将TokenRequest对象转成Flow对象
        step2 执行算法第二步，发送端负载均衡
        step3 执行算法第三步，实现接收端负载均衡
        step4 计算总开销
        '''
        # 清空gpu队列
        for gpu in self.gpus:
            gpu.reset()
        
        # 将目前的req转化成flow对象,注意所有的flow对象都是机间流
        # {(src_gpu, target_node): Flow对象}
        flow_map :dict[(int, int), Flow] = {}
        for req in requests:
            src_node = req.src_gpu // self.config.gpus_per_node
            
            target_nodes = set()
            for target_gpu in set(req.target_gpus):
                target_node = target_gpu // self.config.gpus_per_node
                if src_node != target_node:
                    target_nodes.add(target_node)
                else:
                    if req.src_gpu != target_gpu:
                        self.gpus[req.src_gpu].intra_tx += 1
                        self.gpus[target_gpu].intra_rx += 1

            for target_node in target_nodes:
                key = (req.src_gpu, target_node)
                if key not in flow_map.keys():
                    flow_map[key] = Flow(
                        flow_id=len(flow_map), 
                        src_gpu=req.src_gpu, 
                        target_node=target_node, 
                        size=0
                    )
                flow_map[key].requests.append(req)
                flow_map[key].size += 1
        
        all_flows = list(flow_map.values())

        # 初始化发送端
        gpu_send_queue :defaultdict[int, list[Flow]]= defaultdict(list)
        gpu_send_load = np.zeros((self.config.num_nodes, self.config.gpus_per_node), dtype=int)

        # 发送端负载均衡
        for flow in all_flows:
            src_rank = flow.src_gpu % self.config.gpus_per_node
            src_node = flow.src_gpu // self.config.gpus_per_node

            gpu_send_queue[flow.src_gpu].append(flow)
            gpu_send_load[src_node][src_rank] += flow.size
        
        for node_id in range(self.config.num_nodes):
            current_load = gpu_send_load[node_id]

            max_iter = 100
            for _ in range(max_iter):
                g_h = np.argmax(current_load)
                g_c = np.argmin(current_load)

                hot_gpu_id = self.get_gpu_id(node_id, g_h)
                cold_gpu_id = self.get_gpu_id(node_id, g_c)

                max_load = current_load[g_h]
                min_load = current_load[g_c]
                
                if max_load <= min_load + 1:
                    break

                best_flow = None
                best_gain = 0

                candidate_flows = gpu_send_queue[hot_gpu_id]
                for flow in candidate_flows:
                    copy_current_load = np.copy(current_load)

                    copy_current_load[g_h] -= flow.size
                    copy_current_load[g_c] += flow.size

                    new_bottelneck = np.max(copy_current_load)
                    
                    if new_bottelneck < max_load:
                        gain = max_load - new_bottelneck
                        if gain > best_gain:
                            best_gain = gain
                            best_flow = flow
                
                if best_flow:
                    best_flow.sender = cold_gpu_id
                    gpu_send_queue[hot_gpu_id].remove(best_flow)
                    gpu_send_queue[cold_gpu_id].append(best_flow)
                    current_load[g_h] -= best_flow.size
                    current_load[g_c] += best_flow.size
                else:
                    break
        
        # 初始化接收端
        gpu_recv_queue :defaultdict[int, list[Flow]] = defaultdict(list)
        gpu_recv_load = np.zeros((self.config.num_nodes, self.config.gpus_per_node), dtype=int)

        for flow in all_flows:
            target_node = flow.target_node
            
            if self.config.init_pattern == "deepEP":
                sender_rank = flow.sender % self.config.gpus_per_node
                receiver = self.get_gpu_id(target_node, sender_rank)
            else:
                receiver = flow.requests[0].target_gpus[0]
            
            flow.receiver = receiver
            receiver_rank = receiver % self.config.gpus_per_node
            gpu_recv_queue[flow.receiver].append(flow)
            gpu_recv_load[target_node][receiver_rank] += flow.size
        
        # 贪心迭代
        for node_id in range(self.config.num_nodes):
            current_load = gpu_recv_load[node_id]

            max_iter = 100
            for _ in range(max_iter):
                g_h = np.argmax(current_load)
                g_c = np.argmin(current_load)

                hot_gpu_id = self.get_gpu_id(node_id, g_h)
                cold_gpu_id = self.get_gpu_id(node_id, g_c)

                max_load = current_load[g_h]
                min_load = current_load[g_c]

                if max_load <= min_load + 1:
                    break

                best_flow = None
                best_gain = 0

                for flow in gpu_recv_queue[hot_gpu_id]:
                    copy_current_load = np.copy(current_load)
                    copy_current_load[g_h] -= flow.size
                    copy_current_load[g_c] += flow.size

                    new_bottelneck = np.max(copy_current_load)
                    
                    if new_bottelneck < max_load:
                        gain = max_load - new_bottelneck
                        if gain > best_gain:
                            best_gain = gain
                            best_flow = flow
                
                if best_flow:
                    best_flow.receiver = cold_gpu_id

                    gpu_recv_queue[hot_gpu_id].remove(best_flow)
                    gpu_recv_queue[cold_gpu_id].append(best_flow)

                    current_load[g_h] -= best_flow.size
                    current_load[g_c] += best_flow.size
                else:
                    break
        
        total_inter_tokens = 0
        for flow in all_flows:
            # 机间部分
            self.gpus[flow.sender].inter_tx += flow.size
            self.gpus[flow.receiver].inter_rx += flow.size
            total_inter_tokens += flow.size

            # 源主机offload
            if flow.src_gpu != flow.sender:
                self.gpus[flow.src_gpu].intra_tx += flow.size
                self.gpus[flow.sender].intra_rx += flow.size
            
            # 目的主机分发
            for req in flow.requests:
                for target_gpu in set(req.target_gpus):
                    t_node = target_gpu // self.config.gpus_per_node
                    if t_node == flow.target_node:
                        if flow.receiver != target_gpu:
                            self.gpus[flow.receiver].intra_tx += 1
                            self.gpus[target_gpu].intra_rx += 1
        
        # 计算瓶颈时间
        max_inter_load = 0
        max_intra_load = 0
        
        for gpu in self.gpus:
            max_inter_load = max(max_inter_load, gpu.inter_tx, gpu.inter_rx)
            max_intra_load = max(max_intra_load, gpu.intra_tx, gpu.intra_rx)
        
        time_inter = (max_inter_load * self.config.token_size) / self.config.bw_inter
        time_intra = (max_intra_load * self.config.token_size) / self.config.bw_intra

        analyze_post_workload(self.gpus, "ours-coarse.png")

        layer_time = max(time_inter, time_intra)

        return layer_time, total_inter_tokens * self.config.token_size

if __name__ == "__main__":
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    simulator = OurSimulatorCoarse(config, cluster)
    simulator.run_simulation(args.workload_output_dir)