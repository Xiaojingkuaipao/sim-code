import numpy as np
import dataclasses
from typing import List, Dict, Set
from collections import defaultdict

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================

@dataclasses.dataclass
class SimConfig:
    # 硬件参数
    num_nodes: int = 4              # 节点数量
    gpus_per_node: int = 8          # 每个节点的 GPU 数量
    experts_per_gpu: int = 1        # 每个 GPU 的专家数量
    
    # 带宽参数 (GB/s)
    # Inter-node: 400Gbps ~ 50 GB/s
    # Intra-node: NVLink ~ 400 GB/s
    bw_inter: float = 50.0          
    bw_intra: float = 400.0         
    
    # 负载参数
    num_tokens_total: int = 65536   # 总 Token 数量 (Batch Size * Seq Len)
    token_size_kb: float = 4.0      # 每个 Token 发往 1 个专家的数据量 (KB)
    top_k: int = 2                  # Top-k 路由
    zipf_alpha: float = 1.2         # Zipf 分布参数 (越接近 1 越偏斜)

    @property
    def total_gpus(self):
        return self.num_nodes * self.gpus_per_node

    @property
    def total_experts(self):
        return self.total_gpus * self.experts_per_gpu

# ==========================================
# 2. 数据结构 (Data Structures)
# ==========================================

@dataclasses.dataclass
class TokenRequest:
    token_id: int
    src_gpu: int
    target_experts: List[int]
    target_gpus: List[int] = dataclasses.field(default_factory=list)

class Cluster:
    def __init__(self, config: SimConfig):
        self.cfg = config
        # 专家映射表: expert_id -> gpu_id
        self.expert_to_gpu = {}
        for gpu_id in range(config.total_gpus):
            for e in range(config.experts_per_gpu):
                eid = gpu_id * config.experts_per_gpu + e
                self.expert_to_gpu[eid] = gpu_id

    def get_node(self, gpu_id: int) -> int:
        return gpu_id // self.cfg.gpus_per_node

    def resolve_targets(self, req: TokenRequest):
        """将专家 ID 转换为 GPU ID"""
        req.target_gpus = [self.expert_to_gpu[eid] for eid in req.target_experts]

# ==========================================
# 3. 负载生成器 (Zipf Workload Generator)
# ==========================================

def generate_workload(config: SimConfig, cluster: Cluster) -> List[TokenRequest]:
    print(f"Generating Workload: {config.num_tokens_total} tokens, Zipf alpha={config.zipf_alpha}...")
    
    np.random.seed(42) # 固定随机种子以便复现
    requests = []
    
    # 1. 生成源 GPU (假设 Token 均匀分布在所有 GPU 上)
    src_gpus = np.random.randint(0, config.total_gpus, config.num_tokens_total)
    
    # 2. 生成目标专家 (Zipf 分布)
    # Zipf 生成的是 [1, inf)，我们需要映射到 [0, total_experts-1]
    # 采样数量稍微多一点，以防去重后不足 top_k
    raw_samples = np.random.zipf(config.zipf_alpha, config.num_tokens_total * config.top_k * 2)
    sample_idx = 0
    
    for i in range(config.num_tokens_total):
        # 为每个 Token 选取 Top-k 个不重复的专家
        chosen_experts = set()
        while len(chosen_experts) < config.top_k:
            # 简单的取模哈希，将 Zipf 长尾映射到专家 ID
            eid = (raw_samples[sample_idx] - 1) % config.total_experts
            chosen_experts.add(eid)
            sample_idx += 1
            if sample_idx >= len(raw_samples): # 如果用完了重新生成
                raw_samples = np.random.zipf(config.zipf_alpha, config.num_tokens_total * config.top_k)
                sample_idx = 0
        
        req = TokenRequest(
            token_id=i,
            src_gpu=src_gpus[i],
            target_experts=list(chosen_experts)
        )
        cluster.resolve_targets(req)
        requests.append(req)
        
    return requests

# ==========================================
# 4. 仿真器核心 (Simulator)
# ==========================================

class Simulator:
    def __init__(self, config: SimConfig, cluster: Cluster, requests: List[TokenRequest]):
        self.cfg = config
        self.cluster = cluster
        self.requests = requests
        
        # 单位换算: KB -> GB
        self.token_size_gb = self.cfg.token_size_kb / (1024 * 1024)

    def run_deepep(self):
        """
        DeepEP: Source Aggregation -> Peer GPU -> Dest Node
        瓶颈: 机间带宽最忙的那个 GPU 网卡
        """
        # 记录每个 GPU 需要通过机间网络发送的数据量 (GB)
        nic_inter_vol = defaultdict(float)
        
        for req in self.requests:
            src_node = self.cluster.get_node(req.src_gpu)
            
            # 1. 识别目标节点 (去重实现聚合)
            target_nodes = set()
            for tgt_gpu in req.target_gpus:
                tgt_node = self.cluster.get_node(tgt_gpu)
                if tgt_node != src_node:
                    target_nodes.add(tgt_node)
            
            # 2. 计算流量: 每个目标节点只发 1 份
            for _ in target_nodes:
                nic_inter_vol[req.src_gpu] += self.token_size_gb
        
        # 3. 计算时间
        if not nic_inter_vol:
            max_vol = 0
        else:
            max_vol = max(nic_inter_vol.values())
        
        # Time = Max_Volume / Bandwidth
        latency_us = (max_vol / self.cfg.bw_inter) * 1e6 # 转换为微秒
        
        return {
            "name": "DeepEP",
            "max_volume_gb": max_vol,
            "latency_us": latency_us,
            "bottleneck_gpu": max(nic_inter_vol, key=nic_inter_vol.get) if nic_inter_vol else -1
        }

    def run_fast(self):
        """
        FAST: Intra-node Balance -> Inter-node Transfer
        机制: 
          1. 统计 Node 级别的原始流量 (不聚合, Flow 粒度)
          2. 平均分配给 Node 内的所有 GPU
          3. 加上机内重分布的开销
        """
        # Node 粒度的总流量 (GB)
        node_raw_vol = defaultdict(float)
        # GPU 粒度的原始流量 (用于计算机内重分布开销)
        gpu_raw_vol = defaultdict(float)

        for req in self.requests:
            src_node = self.cluster.get_node(req.src_gpu)
            
            # 1. 计算流量: 不聚合, 按目标 GPU 数量算 (Raw Flows)
            inter_count = 0
            for tgt_gpu in req.target_gpus:
                tgt_node = self.cluster.get_node(tgt_gpu)
                if tgt_node != src_node:
                    inter_count += 1
            
            vol = inter_count * self.token_size_gb
            node_raw_vol[src_node] += vol
            gpu_raw_vol[req.src_gpu] += vol

        # 2. 计算机间传输时间 (均衡后)
        # 每个 Node 的流量平分给 8 个 GPU
        max_node_vol = max(node_raw_vol.values()) if node_raw_vol else 0
        balanced_vol_per_gpu = max_node_vol / self.cfg.gpus_per_node
        time_inter = balanced_vol_per_gpu / self.cfg.bw_inter

        # 3. 计算机内重分布时间 (Intra-node Redistribution)
        # 简单估算: 每个 GPU 需要把多出平均值的部分移出去，或者接收不足的部分
        # 取 Node 内所有 GPU 偏离平均值的最大值作为瓶颈
        max_intra_shift = 0
        for node in range(self.cfg.num_nodes):
            avg = node_raw_vol[node] / self.cfg.gpus_per_node
            for g in range(self.cfg.gpus_per_node):
                gpu_id = node * self.cfg.gpus_per_node + g
                diff = abs(gpu_raw_vol[gpu_id] - avg)
                # 实际上数据是流动的，这里取 diff 作为近似传输量
                # 严谨的算法可以取 sum(diff)/2，但作为瓶颈估算取 max(diff) 比较安全
                max_intra_shift = max(max_intra_shift, diff)
        
        time_intra = max_intra_shift / self.cfg.bw_intra

        total_latency_us = (time_inter + time_intra) * 1e6

        return {
            "name": "FAST",
            "max_volume_gb": balanced_vol_per_gpu, # 均衡后的单卡流量
            "latency_us": total_latency_us,
            "breakdown": f"Inter: {time_inter*1e6:.1f}us, Intra: {time_intra*1e6:.1f}us"
        }

# ==========================================
# 5. 运行脚本
# ==========================================

if __name__ == "__main__":
    # 场景配置
    cfg = SimConfig(
        num_nodes=4, 
        gpus_per_node=8,
        num_tokens_total=2**16, # 65536 tokens
        zipf_alpha=1.1,         # 1.1 非常偏斜, 1.5 中等偏斜, 3.0 接近均匀
        bw_inter=50.0,          # 50 GB/s
        top_k=2
    )
    
    cluster = Cluster(cfg)
    
    # 1. 生成负载
    requests = generate_workload(cfg, cluster)
    
    # 2. 运行仿真
    sim = Simulator(cfg, cluster, requests)
    
    res_deepep = sim.run_deepep()
    res_fast = sim.run_fast()
    
    # 3. 打印结果
    print("\n" + "="*50)
    print(f"Simulation Results (Skewness Alpha = {cfg.zipf_alpha})")
    print("="*50)
    
    print(f"{'Algorithm':<10} | {'Max NIC Vol (GB)':<18} | {'Latency (us)':<15} | {'Note'}")
    print("-" * 60)
    
    print(f"{res_deepep['name']:<10} | "
          f"{res_deepep['max_volume_gb']:<18.4f} | "
          f"{res_deepep['latency_us']:<15.2f} | "
          f"Bottleneck GPU: {res_deepep['bottleneck_gpu']}")
          
    print(f"{res_fast['name']:<10} | "
          f"{res_fast['max_volume_gb']:<18.4f} | "
          f"{res_fast['latency_us']:<15.2f} | "
          f"{res_fast['breakdown']}")

    # 简单分析
    speedup = res_deepep['latency_us'] / res_fast['latency_us']
    if speedup < 1:
        print(f"\nConclusion: FAST is {1/speedup:.2f}x faster than DeepEP.")
    else:
        print(f"\nConclusion: DeepEP is {speedup:.2f}x faster than FAST.")
        
    # print("\n[Analysis]")
    # print("DeepEP suffers from stragglers (one GPU sends too much).")
    # print("FAST suffers from traffic inflation (no aggregation), but has perfect balance.")
    # print("If skew is extreme, DeepEP's bottleneck is huge -> FAST wins.")
    # print("If skew is mild, FAST's inflation penalty > DeepEP's imbalance -> DeepEP wins.")