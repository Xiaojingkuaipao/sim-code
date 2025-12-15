import argparse
from .common import SimConfig, get_args, get_config

def main():
    args = get_args()

    # 配置Simulation 参数
    config = get_config(args=args)