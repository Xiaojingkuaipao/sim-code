# 负载生成说明文档

## 1. Zipf 分布

### 1.1 设计意图

Zipf 分布是模拟长尾分布（Long-tail Distribution）的经典模型。用于模拟**极少数专家承担绝大多数负载**的极端热点场景。

### 1.2 数学原理

Zipf 分布的概率质量函数（PMF）定义为：
$$ P(k) = \frac{1/k^\alpha}{\sum_{n=1}^{N} (1/n^\alpha)} $$
其中：

- $k$ 是专家 id，$k \in \{1, 2, \dots, N\}$。
- $N$ 是专家总数 (`total_experts`)。
- $\alpha$ 是偏斜参数 (`zipf_alpha`)。$\alpha$ 越大，分布越陡峭，热点越集中；$\alpha$ 越接近 0，分布越趋于均匀。

### 1.3 生成逻辑

1.  **独立采样**：对于每个 Token，使用 `numpy.random.zipf(a)` 生成一个随机整数 $X$。
2.  **哈希映射**：由于 Zipf 生成的整数范围是 $[1, \infty)$，我们需要将其映射到有限的专家 ID 空间 $[0, N-1]$。采用取模映射法：
    $$ \text{Expert ID} = (X - 1) \pmod N $$
    这种映射保留了 Zipf 分布的高频特性（例如 $X=1, X=N+1$ 都会映射到 Expert 0，使其成为超级热点）。
3.  **Top-K 去重**：如果采样的 $K$ 个专家有重复，则进行重新采样或随机补全，确保 Top-K 专家的唯一性。

---

## 2. Dirichlet 分布

### 2.1 设计意图

虽然 Zipf 能模拟极端热点，但它倾向于生成单一的静态热点（Rank 1 总是最热）。现实中的 MoE 负载往往呈现**多热点**特征，即可能有多个不同的热点专家，且热点分布具有随机性。Dirichlet 分布用于模拟这种**多热点且热点位置随机**的场景。

### 2.2 数学原理

Dirichlet 分布用于生成一个概率向量 $\mathbf{P} = [p_0, p_1, \dots, p_{N-1}]$，满足 $\sum p_i = 1$。
其概率密度函数为：
$$ f(\mathbf{P}; \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{i=0}^{N-1} p_i^{\alpha_i - 1} $$
其中 $\boldsymbol{\alpha} = [\alpha, \alpha, \dots, \alpha]$ 是浓度参数向量。

- **$\alpha < 1$**：生成的概率向量 $\mathbf{P}$ 是**稀疏**的，即大部分概率质量集中在少数几个分量上（模拟多热点）。
- **$\alpha \gg 1$**：生成的概率向量趋向于均匀分布 $\mathbf{P} \approx [1/N, \dots, 1/N]$。
- 若所有 $\alpha_i=1$：均匀分布在单纯形上；
- 若所有$\alpha_i < 1$（如 0.1）：倾向于生成**稀疏向量**（某些$p_i$接近 1，其余接近 0）；
- 若所有 $\alpha_i > 1$（如 10）：倾向于生成**均匀向量**（各$p_i$接近 $1 / K$）；

### 2.3 生成逻辑

1.  **概率向量生成**：采样一次 Dirichlet 分布，生成该层的全局专家热度向量 $\mathbf{P}$。
2.  **加权采样**：对于该层的所有 Token，基于 $\mathbf{P}$ 进行加权无放回采样，选出 Top-K 个专家

---

## 3. 自定义热点生成器 

### 3.1 设计意图

精确控制热点的数量、位置以及热点承担的流量比例。例如，测试算法在“4 个热点承担 80%流量”与“8 个热点承担 50%流量”下的性能差异。这种**确定性偏斜**是 Zipf 和 Dirichlet 难以精确复现的。

### 3.2 数学原理

该方法采用**分组概率分配**策略，将专家集划分为热点组 $S_{hot}$ 和冷门组 $S_{cold}$。

**Step 1: 组间流量分配**
设定热点流量占比为 $\beta$ (`hot_traffic_ratio`)。

- 热点组总概率质量：$M_{hot} = \beta$
- 冷门组总概率质量：$M_{cold} = 1 - \beta$

**Step 2: 组内概率分配**
为了模拟组内部的随机性（避免组内完全均等），在组内再次应用 Dirichlet 分布。

- 对于热点组 $S_{hot}$，生成随机权重向量 $\mathbf{w}_h$，满足 $\sum \mathbf{w}_h = 1$。
- 对于冷门组 $S_{cold}$，生成随机权重向量 $\mathbf{w}_c$，满足 $\sum \mathbf{w}_c = 1$。

**Step 3: 最终概率合成**
任意专家 $i$ 的被选概率 $P_i$ 计算如下：
$$ P_i = \begin{cases} \beta \cdot w_{h, i} & \text{if } i \in S*{hot} \\ (1 - \beta) \cdot w_{c, i} & \text{if } i \in S\_{cold} \end{cases} $$

### 3.3 生成逻辑

1.  用户输入指定的热点专家列表 $S_{hot}$ 和偏斜度 $\beta$。
2.  计算上述最终概率向量 $\mathbf{P}$。
3.  基于 $\mathbf{P}$ 对所有 Token 进行批量加权采样，生成目标专家序列。
