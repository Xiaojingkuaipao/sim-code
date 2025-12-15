# 仿真整体配置参数
class SimConfig:
    def __init__(
            self, 
            num_nodes: int=4, 
            gpus_per_node: int=8, 
            experts_per_gpu: int=2,
            bw_inter: float=50.0,
            bw_intra: float=400.0,
            seq_len: int=2048,
            batch_size: int=32,
            token_size_kb : float=4.0,
            top_k: int=2,
            zipf_alpha: float=1.2
    ):
        # 集群参数
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.experts_per_gpu = experts_per_gpu

        # 带宽参数 GB/s
        self.bw_inter = bw_inter
        self.bw_intra = bw_intra

        # 模型负载参数
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_size_kb = token_size_kb # 一个token的大小，单位KB
        self.top_k = top_k
        self.zipf_alpha = zipf_alpha
        self.num_tokens_total = seq_len * batch_size
    
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
        target_gpus:list[int] = None
    ):
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