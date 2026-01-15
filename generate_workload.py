import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from common import *
from analyze_workload import analyze_pkl_workload

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

if __name__ == "__main__":
    print("Test Generate Workload")
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    hot_experts = [0, 4, 18, 28, 36, 40, 54, 63]
    hot_traffic_ratio = 0.5
    # generate_workload_zipf(config, cluster, args.workload_output_dir)
    # generate_workload_dirichlet(config, cluster, args.workload_output_dir)
    generate_workload_custom(config, cluster, args.workload_output_dir, hot_experts, hot_traffic_ratio)
    analyze_pkl_workload(args.workload_output_dir, config.total_gpus)