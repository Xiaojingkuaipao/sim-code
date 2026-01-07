import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from common import *
from analyze_workload import analyze_pkl_workload

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


if __name__ == "__main__":
    print("Test Generate Workload")
    args = get_args()
    config = get_config(args)
    cluster = Cluster(config)
    generate_workload(config, cluster, args.workload_output_dir)
    analyze_pkl_workload(args.workload_output_dir, config.total_gpus)