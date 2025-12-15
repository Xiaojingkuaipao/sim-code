import argparse
from .common import SimConfig

def main():
    parser = argparse.ArgumentParser(description="Simulation Experiment")
    parser.add_argument("--num_nodes", default=4, type=int)
    parser.add_argument("--gpus_per_node", default=8, type=int)
    parser.add_argument("--experts_per_gpu", default=2, type=int)
    parser.add_argument("--bw_inter", default=50.0, type=float)
    parser.add_argument("--bw_intra", default=400.0, type=float)
    parser.add_argument("--seq_len", default=2048, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--token_size", default=4.0, type=float)
    parser.add_argument("--topk", default=2, type=int)
    parser.add_argument("--zipf_alpha", default=1.2, type=float)

    args = parser.parse_args()

    # 配置Simulation 参数
    config = SimConfig(
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        experts_per_gpu=args.experts_per_gpu,
        bw_inter=args.bw_inter,
        bw_intra=args.bw_intra,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        token_size_kb=args.token_size,
        top_k=args.topk,
        zipf_alpha=args.zipf_alpha
    )