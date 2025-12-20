# 仿真整体配置参数
# 带宽和token的大小均使用KB 千字节作为单位
# 定义一些通用的工具
import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm


DTYPE_SIZE = {
    "float16": 2,
    "float32": 4,
    "float64": 8
}

class SimConfig:
    def __init__(
            self, 
            num_nodes: int=4, 
            gpus_per_node: int=8, 
            experts_per_gpu: int=2,
            bw_inter: float=50.0,
            bw_intra: float=900.0,
            seq_len: int=2048,
            batch_size: int=32,
            embed_size: int=2048,
            dtype: str="float32",
            top_k: int=2,
            zipf_alpha: float=1.2,
            num_layers: int=12,
            iter_num: int=10
    ):
        '''
        :param num_nodes: 一共有多少个节点
        :param gpus_per_node: 一个节点有多少个gpu
        :param experts_per_gpu: 一个gpu部署多少个expert
        :param bw_inter: inter-node带宽，GB为单位，后续转换为KB
        :param bw_intra: intra-node带宽，GB为单位，后续转换为KB
        :param seq_len: 序列长度，训练参数
        :param batch_size: 略
        :param embed_size: 略
        :param dtype: 数据类型，决定一个token多大
        :param top_k: 路由类型
        :param zipf_alpha: 偏斜程度
        :param num_layers: 一个forward进行多少次All-to-All通信
        :param iter_num: 一个数据集全部训练一轮需要的迭代轮数
        '''
        # 集群参数
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.experts_per_gpu = experts_per_gpu

        # 带宽参数 KB 千字节
        self.bw_inter = bw_inter * 1024 * 1024
        self.bw_intra = bw_intra * 1024 * 1024

        # 模型负载参数
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.embed_size = embed_size

        assert dtype in DTYPE_SIZE.keys(), f"Ecpected dtype:{DTYPE_SIZE.keys()}, but got{dtype}"

        data_size = DTYPE_SIZE[dtype]
        self.dtype = dtype
        self.token_size = embed_size * data_size / 1024 # 一个token的大小，单位KB
        self.top_k = top_k
        self.zipf_alpha = zipf_alpha
        self.num_tokens = seq_len * batch_size
        self.iter_num = iter_num
        self.dataset_toknes = self.num_tokens * self.iter_num
        self.num_layers = num_layers
    
    @property
    def total_gpus(self):
        return self.gpus_per_node * self.num_nodes
    
    @property
    def total_experts(self):
        return self.total_gpus * self.experts_per_gpu
    

# 定义token路由结构
class TokenRequest:
    def __init__(
        self,
        token_id: int,
        src_gpu: int,
        target_experts: list[int],
        target_gpus: list[int] = None
    ):
        '''
        :param token_id: token编号
        :param src_gpu: 源GPU编号
        :param target_experts: 目的expert编号列表
        :param target_gpus: 目的GPU列表
        '''
        self.token_id = token_id
        self.src_gpu = src_gpu
        self.target_experts = target_experts
        if target_gpus is None:
            target_gpus = []
        self.target_gpus = target_gpus

class Cluster:
    def __init__(self, config: SimConfig):
        self.config = config
        # expert_id -> gpu_id
        self.expert_to_gpu = {}
        for gpu_id in range(config.total_gpus):
            for i in range(config.experts_per_gpu):
                expert_id = gpu_id * config.experts_per_gpu + i
                self.expert_to_gpu[expert_id] = gpu_id
    
    def resolve_targets(self, req:TokenRequest):
        '''
        将给定token的target_expert转换为gpu
        '''
        req.target_gpus = [self.expert_to_gpu[eid] for eid in req.target_experts]


class GPU:
    def __init__(self, global_id, node_id, local_rank):
        self.id = global_id
        self.node_id = node_id
        self.local_rank = local_rank

        self.inter_tx = 0
        self.inter_rx = 0
        self.intra_tx = 0
        self.intra_rx = 0
    
    def reset(self):
        self.inter_tx = 0
        self.inter_rx = 0
        self.intra_tx = 0
        self.intra_rx = 0
        

def generate_workload(config: SimConfig, cluster: Cluster, output_dir):
    '''
    生成config.iter_num个需求文件保存到output_dir中
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_requests_count = 0
    print(f"Start Generate Wrokloads......")
    print(f" - Iter Num: {config.iter_num}")
    print(f" - Layer Num: {config.num_layers}")
    print(f" - Token Num Per Iter: {config.num_tokens}")
    print(f" - Zipf Alpha: {config.zipf_alpha}")

    tokens_per_gpu = config.num_tokens // config.total_gpus
    src_gpu_base = np.repeat(np.arange(0, config.total_gpus), tokens_per_gpu)

    # 如果没除尽就把剩下的一些token随机分配给GPU
    if len(src_gpu_base) < config.num_tokens:
        diff = config.num_tokens - len(src_gpu_base)
        src_gpu_base = np.concatenate([src_gpu_base, np.random.randint(0, config.total_gpus, diff)])

    for iter_idx in range(config.iter_num):

        iter_data = []
        
        print(f"Generating Iter:{iter_idx}/{config.iter_num - 1}...")
        for layer_id in tqdm(range(config.num_layers), desc="Layers"):
            layer_request = []

            # zipf [1, inf) -> [0, config.total_experts - 1]
            raw_samples = np.random.zipf(a=config.zipf_alpha, size=(config.num_tokens, config.top_k))
            target_experts_matrix = (raw_samples - 1) % config.total_experts

            for token_id in range(config.num_tokens):
                src_gpu = src_gpu_base[token_id]

                target_experts = target_experts_matrix[token_id].tolist()

                # 如果目标专家数小于topk，也就是一个token有重复的目标专家，是不可能的
                if len(set(target_experts)) < config.top_k:
                    unique_experts = list(set(target_experts))
                    while len(set(unique_experts)) < config.top_k:
                        new_expert = np.random.randint(0, config.total_experts)
                        if new_expert not in unique_experts:
                            unique_experts.append(new_expert)
                    target_experts = unique_experts
                
                req = TokenRequest(
                    token_id=token_id, 
                    src_gpu=src_gpu, 
                    target_experts=target_experts
                )

                cluster.resolve_targets(req=req)
                layer_request.append(req)
            
            iter_data.append(layer_request)
            total_requests_count += len(layer_request)
        
        file_path = os.path.join(output_dir, f"iter_{iter_idx}_requests.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(iter_data, f)
        
        del iter_data
    
    print(f"Workload Generation Complete!")
    print(f"Generated {total_requests_count} token requests in {output_dir}")


def get_args():
    parser = argparse.ArgumentParser(description="Simulation Experiment")
    parser.add_argument("--num_nodes", default=4, type=int)
    parser.add_argument("--gpus_per_node", default=8, type=int)
    parser.add_argument("--experts_per_gpu", default=2, type=int)
    parser.add_argument("--bw_inter", default=50.0, type=float)
    parser.add_argument("--bw_intra", default=400.0, type=float)
    parser.add_argument("--seq_len", default=2048, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--embed_size", default=2048, type=int)
    parser.add_argument("--dtype", default="float16", type=str)
    parser.add_argument("--topk", default=4, type=int)
    parser.add_argument("--zipf_alpha", default=1.2, type=float)
    parser.add_argument("--num_layers", default=12, type=int)
    parser.add_argument("--iter_num", default=10, type=int)
    parser.add_argument("--workload_output_dir", default="./workload", type=str)
    return parser.parse_args()

def get_config(args) -> SimConfig:
    config = SimConfig(
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        experts_per_gpu=args.experts_per_gpu,
        bw_inter=args.bw_inter,
        bw_intra=args.bw_intra,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        embed_size=args.embed_size,
        dtype=args.dtype,
        top_k=args.topk,
        zipf_alpha=args.zipf_alpha,
        num_layers=args.num_layers,
        iter_num=args.iter_num
    )
    return config

if __name__ == "__main__":
    print("Test Generate Workload")
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    generate_workload(config, cluster, args.workload_output_dir)