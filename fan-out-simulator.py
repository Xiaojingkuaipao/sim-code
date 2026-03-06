import os
import pickle
import numpy as np
from common import *
from collections import defaultdict, deque
from analyze_workload import analyze_post_workload
import time

# --- 修改 1: 移除 target_node_set，改为直接传入 weight ---
class TokenClass:
    def __init__(self, class_id, src_gpu, weight):
        self.class_id = class_id
        self.src_gpu = src_gpu
        self.weight = weight
        self.tokens :list[TokenRequest] = []
        self.size = 0
        self.sender = src_gpu

class ClassSimulator:
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
        
        print(f"Start Our Simulation (Fan-out Based Class)")
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


    def simulate_layer(self, requests :list[TokenRequest]):

        for gpu in self.gpus:
            gpu.reset()
        
        start = time.perf_counter()
        
        # --- 修改 2: 分类的 Key 变为 (源GPU, 跨机节点数量/权重) ---
        class_map :dict[tuple[int, int], TokenClass] = {}

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
            if not target_nodes:
                continue

            # 此 Token 产生的跨机发送量
            weight = len(target_nodes)
            key = (req.src_gpu, weight)
            
            if key not in class_map:
                class_map[key] = TokenClass(len(class_map), req.src_gpu, weight)
            
            class_map[key].size += 1
            class_map[key].tokens.append(req)
        
        all_classes :list[TokenClass] = list(class_map.values())

        # 发送端负载均衡
        gpu_send_load = np.zeros((self.config.num_nodes, self.config.gpus_per_node), dtype=int)
        gpu_send_queue :defaultdict[int, list[TokenClass]] = defaultdict(list)

        for c in all_classes:
            src_node = c.src_gpu // self.config.gpus_per_node
            src_rank = c.src_gpu % self.config.gpus_per_node

            gpu_send_queue[c.src_gpu].append(c)
            gpu_send_load[src_node][src_rank] += c.size * c.weight
        
        for node_id in range(self.config.num_nodes):
            current_load = gpu_send_load[node_id]

            for _ in range(100):
                g_h = np.argmax(current_load)
                g_c = np.argmin(current_load)

                max_load = current_load[g_h]
                min_load = current_load[g_c]

                if max_load <= min_load + 1:
                    break
                
                hot_gpu_id = self.get_gpu_id(node_id, g_h)
                cold_gpu_id = self.get_gpu_id(node_id, g_c)

                best_k = 0
                best_class = None
                best_gain = 0

                for c in gpu_send_queue[hot_gpu_id]:

                    if c.size == 0: continue

                    diff = max_load - min_load
                    k_ideal = diff // (2 * c.weight)

                    k = min(c.size, k_ideal)

                    if k <= 0: continue

                    current_load_copy = np.copy(current_load)
                    current_load_copy[g_h] -= k * c.weight
                    current_load_copy[g_c] += k * c.weight
                    
                    new_bottleneck = np.max(current_load_copy)

                    if new_bottleneck < max_load:
                        gain = max_load - new_bottleneck
                        if gain > best_gain:
                            best_gain = gain
                            best_k = k
                            best_class = c
                
                if best_k > 0 and best_class:
                    move_tokens = []
                    for _ in range(best_k):
                        move_tokens.append(best_class.tokens.pop())
                    best_class.size -= best_k

                    # --- 迁移逻辑保持不变，创建新 Class 时传入 weight ---
                    new_class = TokenClass(-1, best_class.src_gpu, best_class.weight)
                    new_class.size = best_k
                    new_class.tokens = move_tokens
                    new_class.sender = cold_gpu_id
                    gpu_send_queue[cold_gpu_id].append(new_class)

                    current_load[g_h] -= best_k * best_class.weight
                    current_load[g_c] += best_k * best_class.weight
                else:
                    break
        
        # --- 修改 3: 从 TokenClass 拆解出细粒度的 Flow ---
        # 因为现在一个 TokenClass 里的 token 目标可能不同，我们需要按 token 本身的信息去拆解
        all_flows :list[Flow] = []
        for send_queue in gpu_send_queue.values():
            for c in send_queue:
                if c.size == 0: continue

                for token in c.tokens:
                    # 获取该 Token 的所有目标节点
                    t_nodes = set()
                    for target_gpu in token.target_gpus:
                        t_node = target_gpu // self.config.gpus_per_node
                        # 过滤掉同机流量
                        if t_node != (token.src_gpu // self.config.gpus_per_node):
                            t_nodes.add(t_node)
                    
                    # 针对每个目标节点，生成一个独立的 Flow
                    for target_node in t_nodes:
                        target_gpus = []
                        for target_gpu in token.target_gpus:
                            if target_gpu // self.config.gpus_per_node == target_node:
                                target_gpus.append(target_gpu)
                        
                        flow = Flow(token.token_id, token.src_gpu, target_node, target_gpus)
                        flow.sender = c.sender  # 继承 Step 2 调度好的发送者
                        all_flows.append(flow)
        
        # 接收端负载均衡 (以下逻辑全部保持不变)
        gpu_recv_queue :defaultdict[int, deque[Flow]] = defaultdict(deque)
        gpu_recv_load = np.zeros((self.config.num_nodes, self.config.gpus_per_node), dtype=int)
        
        # 初始化接收队列和接收负载矩阵
        for flow in all_flows:
            target_node = flow.target_node

            if self.config.init_pattern == "deepEP":
                # method1:默认接收者和deepEP相同，为同号卡
                sender_rank = flow.sender % self.config.gpus_per_node
                receiver = target_node * self.config.gpus_per_node + sender_rank
            else:
                # method2:默认接收者为flow.target_gpus中的第一个gpu
                receiver = flow.target_gpus[0]

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

                if not gpu_recv_queue[hot_gpu_id]:
                    break

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


        processed_source_tokens = set()
        
        for flow in all_flows:
            # 源端节点中继
            if flow.src_gpu != flow.sender:
                token_key = (flow.src_gpu, flow.sender, flow.token_id)
                if token_key not in processed_source_tokens:
                    self.gpus[flow.src_gpu].intra_tx += 1
                    self.gpus[flow.sender].intra_rx += 1
                    processed_source_tokens.add(token_key)
            
            # 目的端分发
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

        try:
            analyze_post_workload(self.gpus, "our-fan-out.png")
        except NameError:
            pass # 如果没有引入作图函数则跳过
        
        time_inter = (max_inter_load * self.config.token_size) / self.config.bw_inter
        time_intra = (max_intra_load * self.config.token_size) / self.config.bw_intra

        layer_time = max(time_inter, time_intra)
        
        return layer_time, total_inter_token * self.config.token_size

if __name__ == "__main__":
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    simulator = ClassSimulator(config, cluster)
    simulator.run_simulation(args.workload_output_dir)