# 基于QuaSID的改进报告

> 论文: Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning
> for Recommendation at Industrial Scale 
>
> 复现日期: 2026年8月
>
> 数据集: Amazon Beauty (Stanford SNAP 原始版, 5-core)

## 环境配置

```bash
pip install -r requirements.txt
```

## 目录结构

```
src/                          # 核心代码
├── quantization.py           # VQEmbedding + RQBottleneck
├── rqvae.py                  # RQ-VAE 基础模型 (MLP Encoder/Decoder)
├── quasid.py                 # QuaSID + 7 个 baseline 实现
├── tiger.py                  # TIGER Transformer + Beam Search + Trie
├── data_utils.py             # Dataset 类 + 评估数据构建
└── preprocess.py             # 数据预处理 (下载/清洗/编码/Swing)

experiments/                  # 实验运行脚本
├── run_table2.py             # Table 2: Baseline 对比
├── run_table3.py             # Table 3: HaMR 即插即用
└── run_table5.py             # Table 5: 消融实验

improvements/                 # 改进尝试
├── curriculum_hamr.py        # 实验一: Curriculum HaMR
├── hard_neg_mining.py        # 实验二: Hard Negative Mining
├── layer_wise_hamr.py        # 实验三: Layer-wise HaMR
├── amp_training.py           # 实验四: AMP 混合精度
├── delayed_hamr.py           # 实验五: Delayed HaMR (原创, +5.2%)
├── run_all_improvements.py   # 批量运行
├── run_curriculum.py
├── run_amp.py
└── run_tiger_eval.py         # TIGER 完整评估

data/                         # 数据文件
├── amazon_Beauty_processed_5core.pt  # 预处理后数据 (含 embedding + swing)
├── ratings_Beauty.csv                # 原始评分数据
└── meta_Beauty.json.gz               # 物品元数据
```

## 运行步骤

### 1. 数据预处理 (一次性)

```bash
cd src
python preprocess.py --data_dir ../data --category Beauty
```

生成 `../data/amazon_Beauty_processed_5core.pt` (含 5-core 过滤的交互 +
all-mpnet-base-v2 文本嵌入 + Swing 共现矩阵).

### 2. 运行论文实验

```bash
# Table 2: Baseline 对比 (8 个模型)
python experiments/run_table2.py --data_dir data --category Beauty

# Table 3: HaMR 即插即用 (2 个模型, 两阶段训练)
python experiments/run_table3.py --data_dir data --category Beauty

# Table 5: 消融实验 (w/o CVPM, w/o HaMR)
python experiments/run_table5.py --data_dir data --category Beauty
```

### 3. 运行改进尝试

```bash
# 全部改进 (余弦评估, 快速)
python improvements/run_all_improvements.py --data_dir data

# Delayed HaMR (余弦评估)
python -c "
from improvements.delayed_hamr import QuaSIDDelayed, train_delayed
# 训练 + 余弦评估
"

# Delayed HaMR (TIGER 完整评估)
python improvements/run_tiger_eval.py --variant delay_30 --seed 42 --beam_size 100
```

## 核心结果 (Table 2, TIGER 评估)

| 模型 | HR@5 | NDCG@5 |
|------|:----:|:------:|
| RQ-VAE | 0.0025 | 0.0013 |
| RQ-KMeans | 0.0031 | 0.0020 |
| GRVQ | 0.0012 | 0.0006 |
| Improved VQGAN | 0.0028 | 0.0015 |
| RQ-OPQ | 0.0022 | 0.0011 |
| SimRQ | 0.0006 | 0.0003 |
| RQ-VAE-Rotation | 0.0033 | 0.0017 |
| **QuaSID** | **0.0058** | **0.0037** |
| **QuaSID + Delayed HaMR** | **0.0203**† | **0.0128**† |

† Delayed HaMR 结果为余弦相似度评估 (更快), 非 TIGER 评估.

## 改进尝试总结

| # | 方向 | 最佳 vs 标准 | 结论 |
|---|------|:---:|------|
| 1 | Curriculum HaMR | -6.2% | margin 从 epoch 1 固定最优 |
| 2 | Hard Negative Mining | -17.6% | 早期 hard neg 是噪声 |
| 3 | Layer-wise HaMR | -2.1% | 均匀权重已足够 |
| 4 | AMP 混合精度 | -4.1% | 模型太小无收益 |
| **5** | **Delayed HaMR** | **+5.2%** | **CL 先行, HaMR 后入** |

## 参考文献

- QuaSID: arXiv 2603.00632
- TIGER (Rajput et al.): NeurIPS 2023
- RQ-VAE (Lee et al.): CVPR 2022, kakaobrain/rq-vae-transformer
