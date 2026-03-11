import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from common import *
from analyze_workload import *

def generate_workload_zipf(config: SimConfig, cluster: Cluster, output_dir):
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

def generate_workload_dirichlet(config :SimConfig, cluster :Cluster, output_dir):
    '''
    使用dirichlet分布生成工作负载
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_requests_count = 0
    print(f"Start Generate Wrokloads......")
    print(f" - Iter Num: {config.iter_num}")
    print(f" - Layer Num: {config.num_layers}")
    print(f" - Token Num Per Iter: {config.num_tokens}")
    print(f" - Dirichlet Alpha: {config.zipf_alpha}")

    tokens_per_gpu = config.num_tokens // config.total_gpus
    src_gpu_base = np.repeat(np.arange(0, config.total_gpus), tokens_per_gpu)

    if len(src_gpu_base) < config.num_tokens:
        diff = config.num_tokens - len(src_gpu_base)
        src_gpu_base = np.concatenate([src_gpu_base, np.random.randint(0, config.total_gpus, diff)])
    
    for iter_idx in range(config.iter_num):

        iter_data = []
        
        print(f"Generating Iter:{iter_idx}/{config.iter_num - 1}...")
        for layer_id in tqdm(range(config.num_layers), desc="Layers"):
            layer_request = []

            alpha_vec = np.full(config.total_experts, config.zipf_alpha)
            expert_prob = np.random.dirichlet(alpha_vec)

            raw_samples = np.random.choice(
                config.total_experts,
                (config.num_tokens, config.top_k * 2),
                replace=True,
                p=expert_prob
            )

            for token_id in range(config.num_tokens):
                src_gpu = src_gpu_base[token_id]
                candidate_experts = raw_samples[token_id]

                target_experts = []
                for exp in candidate_experts:
                    exp = int(exp)
                    if exp not in target_experts:
                        target_experts.append(exp)
                        if len(target_experts) == config.top_k:
                            break
                
                while len(target_experts) < config.top_k:
                    new_exp = np.random.randint(0, config.total_experts, dtype=int)
                    if new_exp not in target_experts:
                        target_experts.append(new_exp)
                
                req = TokenRequest(
                    token_id=token_id,
                    src_gpu=src_gpu,
                    target_experts=target_experts,
                )

                cluster.resolve_targets(req)
                layer_request.append(req)
        
            iter_data.append(layer_request)
            total_requests_count += len(layer_request)

        with open(os.path.join(output_dir, f"iter_{iter_idx}_requests.pkl"), "wb") as f:
            pickle.dump(iter_data, f)
        
        del iter_data
    
    print(f"Workload Generation Complete!")
    print(f"Generated {total_requests_count} token requests in {output_dir}")

def generate_workload_custom(
        config :SimConfig,
        cluster :Cluster,
        output_dir,
        hot_experts :list[int],
        hot_traffic_ratio :float 
):
    '''
    接收热点专家列表和热点专家组的流量比例，生成不均衡的工作负载
    '''
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not hot_experts:
        raise ValueError("Parameter hot_experts can not be empty")
    
    if hot_traffic_ratio is None:
        raise ValueError("Parameter hot_experts can not be empty")
    
    if len(hot_experts) >= config.total_experts:
        raise ValueError("All experts cannot be hot experts")
    
    if not (0 <= hot_traffic_ratio <= 1):
        raise ValueError("hot_traffic_ratio must between 0 and 1")
    
    total_experts = config.total_experts
    prob_vector = np.zeros(total_experts, dtype=float)

    cold_experts = [i for i in range(total_experts) if i not in hot_experts]

    num_hot_experts = len(hot_experts)
    num_cold_experts = len(cold_experts)

    intra_group_alpha = config.zipf_alpha

    if num_hot_experts > 0:
        hot_internal_weights = np.random.dirichlet(np.full(num_hot_experts, intra_group_alpha))
        prob_vector[hot_experts] = hot_internal_weights * hot_traffic_ratio

    if num_cold_experts > 0:
        cold_internal_weights = np.random.dirichlet(np.full(num_cold_experts, intra_group_alpha))
        prob_vector[cold_experts] = cold_internal_weights * (1.0 - hot_traffic_ratio)
    
    prob_vector = prob_vector / prob_vector.sum()
    
    total_requests_count = 0
    tokens_per_gpu = config.num_tokens // config.total_gpus

    src_gpu_base = np.repeat(np.arange(0, config.total_gpus), tokens_per_gpu)

    if len(src_gpu_base) < config.num_tokens:
        diff = config.num_tokens - len(src_gpu_base)
        src_gpu_base = np.concatenate([src_gpu_base, np.random.randint(0, config.total_gpus, diff)])

    print(f"Start Generate Custom Workloads...")
    print(f" - Hot Ratio: {hot_traffic_ratio}")
    print(f" - Hot Experts: {hot_experts}")

    for iter_idx in range(config.iter_num):
        iter_data = []

        for layer_id in range(config.num_layers):
            layer_request = []

            raw_samples = np.random.choice(
                total_experts,
                (config.num_tokens, config.top_k * 2),
                replace=True,
                p=prob_vector
            )
            
            for token_id in range(config.num_tokens):
                candidate = raw_samples[token_id]
                target_experts = []

                for exp in candidate:
                    exp = int(exp)
                    if exp not in target_experts:
                        target_experts.append(exp)
                        if len(target_experts) == config.top_k:
                            break

                while len(target_experts) < config.top_k:
                    new_exp = np.random.randint(0, total_experts)
                    if new_exp not in target_experts:
                        target_experts.append(new_exp)
                
                req = TokenRequest(
                    token_id,
                    src_gpu_base[token_id],
                    target_experts
                )
                cluster.resolve_targets(req)
                layer_request.append(req)
            
            iter_data.append(layer_request)
            total_requests_count += len(layer_request)
        
        file_path = os.path.join(output_dir, f"iter_{iter_idx}_requests.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(iter_data, f)
        
        del iter_data
    print(f"Workload Generation Complete!")
    print(f"Generated {total_requests_count} token requests in {output_dir}")


def generate_workload_zipf_custom(config: SimConfig, cluster: Cluster, output_dir: str):
    '''
    生成标准的有限域 Zipf 分布负载。
    完美支持 config.zipf_alpha 在 0.0 ~ 1.0 之间的取值。
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_requests_count = 0
    print(f"Start Generate Workloads (Standard Bounded Zipf)...")
    print(f" - Iter Num: {config.iter_num}")
    print(f" - Layer Num: {config.num_layers}")
    print(f" - Token Num Per Iter: {config.num_tokens}")
    print(f" - Zipf Alpha: {config.zipf_alpha} (Supports 0.0 ~ 1.0)")

    # 1. 源 GPU 分布 (均匀分布)
    tokens_per_gpu = config.num_tokens // config.total_gpus
    src_gpu_base = np.repeat(np.arange(0, config.total_gpus), tokens_per_gpu)
    if len(src_gpu_base) < config.num_tokens:
        diff = config.num_tokens - len(src_gpu_base)
        src_gpu_base = np.concatenate([src_gpu_base, np.random.randint(0, config.total_gpus, diff)])

    # =========================================================
    # 核心修改：手动构建 Bounded Zipf 概率分布
    # =========================================================
    N = config.total_experts
    alpha = config.zipf_alpha
    
    # 排名 1 到 N
    ranks = np.arange(1, N + 1)
    
    # 计算权重 1 / (k^alpha)
    # 如果 alpha = 0，weights 全是 1，变成均匀分布
    weights = 1.0 / np.power(ranks, alpha)
    
    # 归一化，得到真正的概率分布向量
    zipf_prob_vector = weights / np.sum(weights)
    # =========================================================

    for iter_idx in range(config.iter_num):
        iter_data =[]
        
        print(f"Generating Iter:{iter_idx}/{config.iter_num - 1}...")
        for layer_id in tqdm(range(config.num_layers), desc="Layers"):
            layer_request =[]

            # 批量采样：使用构建好的 zipf_prob_vector
            raw_samples = np.random.choice(
                N, # 专家 ID 从 0 到 N-1
                size=(config.num_tokens, config.top_k * 2),
                replace=True,
                p=zipf_prob_vector # 严格按照 Zipf 概率采样
            )

            for token_id in range(config.num_tokens):
                src_gpu = src_gpu_base[token_id]
                candidate_experts = raw_samples[token_id]

                target_experts =[]
                for exp in candidate_experts:
                    exp = int(exp)
                    if exp not in target_experts:
                        target_experts.append(exp)
                        if len(target_experts) == config.top_k:
                            break
                
                while len(target_experts) < config.top_k:
                    new_exp = int(np.random.randint(0, N))
                    if new_exp not in target_experts:
                        target_experts.append(new_exp)
                
                req = TokenRequest(
                    token_id=token_id,
                    src_gpu=src_gpu,
                    target_experts=target_experts,
                )

                cluster.resolve_targets(req)
                layer_request.append(req)
            
            iter_data.append(layer_request)
            total_requests_count += len(layer_request)
        
        file_path = os.path.join(output_dir, f"iter_{iter_idx}_requests.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(iter_data, f)
        
        del iter_data
        
    print(f"Workload Generation Complete!")
    print(f"Generated {total_requests_count} token requests in {output_dir}")

# very important
def generate_workload_bimodal_skew(
        config: SimConfig, 
        cluster: Cluster, 
        output_dir: str,  # 负载.pkl输出到的文件夹
        hot_ranks: list[int] = [0, 1],  
        hot_remote_ratio: float = 0.9, 
        cold_remote_ratio: float = 0.1
):
    """
    生成同时具备 [源端Rank偏斜] 和 [目标Zipf偏斜] 的负载。
    """
    # --- 1. 参数验证 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not (0 <= hot_remote_ratio <= 1.0):
        raise ValueError(f"hot_remote_ratio must be in [0, 1], got {hot_remote_ratio}")
    if not (0 <= cold_remote_ratio <= 1.0):
        raise ValueError(f"cold_remote_ratio must be in [0, 1], got {cold_remote_ratio}")
    
    # 验证 hot_ranks 是否在合法范围内
    if any(r < 0 or r >= config.gpus_per_node for r in hot_ranks):
        raise ValueError(f"hot_ranks elements must be in [0, {config.gpus_per_node - 1}]")

    # 验证专家数量是否足够
    # 注意：这里做的是全局检查，更严格的检查在循环内部针对每个节点进行
    if config.total_experts < config.top_k:
        raise ValueError(f"Total experts ({config.total_experts}) must be >= top_k ({config.top_k})")

    print(f"Start Generate Bimodal Skew Workloads...")
    print(f" - Hot Ranks: {hot_ranks} (Remote Ratio: {hot_remote_ratio*100}%)")
    print(f" - Cold Ranks Remote Ratio: {cold_remote_ratio*100}%")
    print(f" - Global Target Zipf Alpha: {config.zipf_alpha}")

    # 每个 GPU 的固定 Token 数量 (严格整除，丢弃余数以保证均匀)
    if config.num_tokens % config.total_gpus != 0:
        print(f"Warning: num_tokens ({config.num_tokens}) is not divisible by total_gpus ({config.total_gpus}). Remainder will be dropped.")
    
    tokens_per_gpu = config.num_tokens // config.total_gpus
    
    # 预先计算全局专家的 Zipf 概率分布 (用于跨机 Token 选择专家)
    ranks = np.arange(1, config.total_experts + 1) 
    weights = 1.0 / np.power(ranks, config.zipf_alpha) # 每个专家的权重 1 / (k^{\alpha})
    zipf_prob_vector = weights / np.sum(weights) # p(k) = (1 / k) / (\sum_{n=1}^{N}{1 / n^{\alpha}})

    total_generated_count = 0

    for iter_idx in range(config.iter_num): # 这个循环相当于没有，因为多层的结果都是一样的，所以默认的config.iter_num=1
        iter_data = []
        print(f"Generating Iter:{iter_idx}/{config.iter_num - 1}...")
        
        # 重置每轮迭代的 token_id 计数器
        current_iter_token_id = 0

        # 这个循环也相当于没有，因为不管多少层输入负载的分布一致，结果差不太多，所以只需要一层即可
        for layer_id in tqdm(range(config.num_layers), desc="Layers"): 
            layer_request = []
            
            # 遍历每一个物理 GPU
            for gpu_id in range(config.total_gpus):
                node_id = gpu_id // config.gpus_per_node
                local_rank = gpu_id % config.gpus_per_node
                
                # 划分本地专家(在同一个node内)和远程专家(在其他node中)
                local_expert_ids = [e for e, g in cluster.expert_to_gpu.items() if g // config.gpus_per_node == node_id]
                local_expert_set = set(local_expert_ids)
                remote_expert_ids_count = config.total_experts - len(local_expert_ids) # 远程专家的数量

                # 检查专家数量是否满足 top_k
                if len(local_expert_ids) < config.top_k:
                     # 如果本地专家不足，无法生成纯本地的不重复 Token
                     # 这里选择抛出异常，或者根据需求改为允许重复
                     raise ValueError(f"Node {node_id} has {len(local_expert_ids)} experts, but top_k is {config.top_k}. Cannot satisfy local request without replacement.")
                
                if remote_expert_ids_count < config.top_k:
                    raise ValueError(f"Node {node_id} has {remote_expert_ids_count} remote experts, but top_k is {config.top_k}. Cannot satisfy remote request.")

                # 根据 Rank 决定跨机比例
                if local_rank in hot_ranks: # 如果当前rank是hot rank(机间发送负载重的rank)
                    num_remote = int(tokens_per_gpu * hot_remote_ratio) # 当前GPU需要跨机发送的token数
                else:
                    num_remote = int(tokens_per_gpu * cold_remote_ratio)
                
                num_local = tokens_per_gpu - num_remote # 目的地在本机内的token数量

                # 1. 生成 Local Tokens
                for _ in range(num_local):
                    # 本地 Token 随机选择本地专家 (replace=False表示无放回抽样，一个token不会选择到同一个专家两次)
                    candidates = np.random.choice(local_expert_ids, config.top_k, replace=False) 
                    target_experts = [int(exp) for exp in candidates] # 本地专家的id列表
                        
                    req = TokenRequest(token_id=current_iter_token_id, src_gpu=gpu_id, target_experts=target_experts)
                    cluster.resolve_targets(req)
                    layer_request.append(req)
                    current_iter_token_id += 1
                    total_generated_count += 1

                # 2. 生成 Remote Tokens (带 Zipf 偏斜)
                for _ in range(num_remote):
                    target_experts = []
                    
                    # 使用 while 循环配合批量采样
                    while len(target_experts) < config.top_k:
                        # 批量采样以提高效率 (2倍 top_k 通常足够)
                        candidates = np.random.choice(
                            config.total_experts,
                            size=config.top_k * 2,
                            p=zipf_prob_vector,
                            replace=True
                        )
                        
                        for exp in candidates:
                            exp = int(exp)
                            # 必须是远程专家 (不在 local_expert_set 中) 且未被选过
                            if exp not in local_expert_set and exp not in target_experts:
                                target_experts.append(exp)
                                if len(target_experts) == config.top_k:
                                    break
                    
                    req = TokenRequest(token_id=current_iter_token_id, src_gpu=gpu_id, target_experts=target_experts)
                    cluster.resolve_targets(req)
                    layer_request.append(req)
                    current_iter_token_id += 1
                    total_generated_count += 1
                    
            # TODO 这个token可能既被发送到本地GPU上,又发到远端GPU上,上述做法有些绝对,后续商榷是否有更好的方法
            iter_data.append(layer_request)
        
        # 保存数据
        file_path = os.path.join(output_dir, f"iter_{iter_idx}_requests.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(iter_data, f)
        
        del iter_data

    print(f"Workload Generation Complete!")
    print(f"Generated {total_generated_count} token requests in {output_dir}")
if __name__ == "__main__":
    print("Test Generate Workload")
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    hot_experts = [0, 4, 18, 28, 36, 40, 54, 63]
    hot_traffic_ratio = 0.5
    
    hot_ranks = [0, 1]
    hot_remote_ratio = 0.4
    cold_remote_ratio = 0.2
    # generate_workload_zipf(config, cluster, args.workload_output_dir)
    # generate_workload_dirichlet(config, cluster, args.workload_output_dir)
    # generate_workload_custom(config, cluster, args.workload_output_dir, hot_experts, hot_traffic_ratio)
    # generate_workload_zipf_custom(config, cluster, args.workload_output_dir)
    generate_workload_bimodal_skew(
        config=config,
        cluster=cluster,
        output_dir=args.workload_output_dir,
        hot_ranks=hot_ranks,
        hot_remote_ratio=hot_remote_ratio,
        cold_remote_ratio=cold_remote_ratio
    )
    analyze_bimodal_skew(
        args.workload_output_dir, config, hot_ranks, hot_remote_ratio, cold_remote_ratio, cluster
    )