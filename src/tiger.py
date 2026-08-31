"""
TIGER: Transformer Index for GEnerative Recommenders
严格遵循 QuaSID 论文附录 A.2 配置

配置:
- 8 层 Transformer: 4 encoder + 4 decoder (论文原始 TIGER 架构)
- 8 attention heads
- d_model = 128
- MLP hidden dim = 512
- max_seq_len = 80 (截断长序列, ~26 个物品)
- dropout = 0.1
- Adam lr=3e-4, wd=1e-5
- Batch size = 256
- SID: L=3 codebooks, K=256 per codebook

Token 词汇表:
  Layer 0 (c1): 0~255
  Layer 1 (c2): 256~511
  Layer 2 (c3): 512~767
  PAD: 768, BOS: 769, EOS: 770
  Total: 771 tokens

参考:
- TIGER paper (Rajput et al., NeurIPS 2023, arxiv 2305.05065)
- QuaSID paper (arxiv 2603.00632)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict


# ==============================================================================
# TIGER Transformer 模型
# ==============================================================================

class TIGER(nn.Module):
    """
    TIGER 生成式检索 Transformer

    Encoder-Decoder 架构, 自回归生成 Semantic ID tokens
    """

    def __init__(self,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 n_codebook=3,
                 n_embed=256,
                 max_seq_len=80):
        super().__init__()

        self.d_model = d_model
        self.n_codebook = n_codebook
        self.n_embed = n_embed
        self.max_seq_len = max_seq_len

        # ── 词汇表 ──
        self.base_vocab_size = n_codebook * n_embed   # 768
        self.pad_token_id = self.base_vocab_size       # 768
        self.bos_token_id = self.base_vocab_size + 1   # 769
        self.eos_token_id = self.base_vocab_size + 2   # 770
        self.vocab_size = self.base_vocab_size + 3     # 771

        # 每个位置允许的 token 范围 (用于 beam search 约束)
        self.register_buffer('layer_offset',
                             torch.tensor([l * n_embed for l in range(n_codebook)]),
                             persistent=False)

        # ── 嵌入层 ──
        self.token_embedding = nn.Embedding(
            self.vocab_size, d_model, padding_idx=self.pad_token_id)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # ── Transformer ──
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )

        # ── 输出投影 ──
        self.output_proj = nn.Linear(d_model, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── SID ⇄ Token 转换 ──

    def encode_sid(self, codes):
        """
        SID codes → token IDs (加层偏移)

        Args:
            codes: [..., L'] SID codes (0~K-1 per layer, can be 1 to n_codebook)
        Returns:
            tokens: [..., L'] token IDs
        """
        L = codes.shape[-1]
        offset = self.layer_offset[:L].view(*([1] * (codes.dim() - 1)), -1)
        return codes + offset

    def decode_sid(self, tokens):
        """
        Token IDs → SID codes (去层偏移)

        Args:
            tokens: [..., L'] token IDs (can be 0 to n_codebook tokens)
        Returns:
            codes: [..., L'] SID codes (0~K-1 per layer)
        """
        if tokens.numel() == 0:
            return tokens
        # Only decode the layers that are actually present
        L = tokens.shape[-1]  # number of tokens in last dim
        offset = self.layer_offset[:L]  # only first L offsets
        offset = offset.view(*([1] * (tokens.dim() - 1)), -1)
        return tokens - offset

    # ── 位置编码 ──

    def _add_position(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.positional_embedding(positions)

    # ── Forward ──

    def forward(self, src, tgt):
        """
        训练前向传播

        Args:
            src: [B, src_len] encoder 输入 token IDs (物品历史 SID 序列)
            tgt: [B, tgt_len] decoder 输入 token IDs (shifted right)
                 tgt = [BOS, c1+0K, c2+1K, c3+2K], 去掉了末尾 EOS
        Returns:
            logits: [B, tgt_len, vocab_size] (含逐层 mask)
        """
        # 截断过长序列
        if src.size(1) > self.max_seq_len:
            src = src[:, -self.max_seq_len:]
        if tgt.size(1) > self.max_seq_len:
            tgt = tgt[:, -self.max_seq_len:]

        # 掩码: 统一用 float (0.0=允许, -inf=屏蔽), 避免与 tgt_mask 类型不匹配
        tgt_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device)
        src_pad_mask = torch.zeros(src.size(0), src.size(1), device=src.device)
        src_pad_mask[src == self.pad_token_id] = float('-inf')
        tgt_pad_mask = torch.zeros(tgt.size(0), tgt.size(1), device=tgt.device)
        tgt_pad_mask[tgt == self.pad_token_id] = float('-inf')

        # 嵌入 + 位置编码
        src_emb = self.dropout(self._add_position(
            self.token_embedding(src)))
        tgt_emb = self.dropout(self._add_position(
            self.token_embedding(tgt)))

        # Transformer
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        logits = self.output_proj(output)  # [B, tgt_len, vocab_size]

        # ── 逐层 token 约束 (与 beam search 保持一致) ──
        # Position 0: 仅允许 c1 (0 ~ K-1)
        # Position 1: 仅允许 c2 (K ~ 2K-1)
        # Position 2: 仅允许 c3 (2K ~ 3K-1)
        # Position 3: 仅允许 EOS
        # 这样训练和推理的 token 空间一致，模型不用浪费容量学习无效 token
        layer_mask = torch.full_like(logits, float('-inf'))
        for pos in range(min(self.n_codebook, tgt_len)):
            layer_mask[:, pos, pos * self.n_embed:(pos + 1) * self.n_embed] = 0
        # 最后一个位置预测 EOS
        if tgt_len > self.n_codebook:
            layer_mask[:, self.n_codebook, self.eos_token_id] = 0
        # BOS 只在第一步出现，不需要单独 mask（tgt 中已无 BOS）

        return logits + layer_mask

    # ── 批量 Encoder (评估优化) ──

    @torch.no_grad()
    def encode_batch(self, src_list):
        """
        批量编码多个源序列

        Args:
            src_list: list of [src_len_i] tensor
        Returns:
            memory_list: list of [1, src_len_i, d_model] tensor
        """
        # 按长度分组，减少 padding 浪费
        memories = []
        # 逐个处理 (简单实现，避免复杂 padding 逻辑)
        for src in src_list:
            if src.size(0) > self.max_seq_len:
                src = src[-self.max_seq_len:]
            src_emb = self.dropout(self._add_position(
                self.token_embedding(src.unsqueeze(0))))
            mem = self.transformer.encoder(src_emb)
            memories.append(mem)
        return memories

    # ── Beam Search 推理 (带 Trie 约束) ──

    @torch.no_grad()
    def beam_search(self, src, beam_size=100, max_len=None,
                    valid_tokens_per_step=None):
        """
        Beam search 生成 SID tokens (单个序列, 带 Trie 约束)

        Args:
            src: [src_len] 单个序列的 encoder 输入
            beam_size: beam 大小
            max_len: 最大生成长度 (默认 n_codebook, 不含 EOS)
            valid_tokens_per_step: list of sets, 每步允许的 token (SID范围 + Trie约束)
                step 0: {c1 that exists in at least one item}
                step 1: {c2+offset that exists for a given c1 prefix}
                step 2: {c3+offset that exists for a given (c1,c2) prefix}
        Returns:
            sequences: [beam_size, max_len] 生成的 token IDs
            scores: [beam_size] log-prob 分数
        """
        if max_len is None:
            max_len = self.n_codebook

        device = src.device

        # Encoder
        if src.size(0) > self.max_seq_len:
            src = src[-self.max_seq_len:]
        src_emb = self.dropout(self._add_position(
            self.token_embedding(src.unsqueeze(0))))
        memory = self.transformer.encoder(src_emb)  # [1, src_len, d_model]

        # Beam search (批量化)
        beams = [(torch.tensor([[self.bos_token_id]], device=device), 0.0, False)]

        for step in range(max_len):
            active_beams = [(tok, sc, d) for tok, sc, d in beams if not d]
            done_beams = [(tok, sc, d) for tok, sc, d in beams if d]

            if not active_beams:
                break

            # 批量化 decoder
            tgt_batch = torch.cat([t for t, _, _ in active_beams], dim=0)
            num_active = tgt_batch.size(0)

            tgt_emb = self.dropout(self._add_position(
                self.token_embedding(tgt_batch)))
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                tgt_batch.size(1), device=device)
            mem_batch = memory.expand(num_active, -1, -1)

            output = self.transformer.decoder(
                tgt_emb, mem_batch, tgt_mask=tgt_mask)
            logits = self.output_proj(output[:, -1, :])  # [B, vocab]

            # 限制到当前 layer 的 token 范围
            layer_start = step * self.n_embed
            layer_end = (step + 1) * self.n_embed
            mask = torch.full_like(logits, float('-inf'))
            mask[:, layer_start:layer_end] = logits[:, layer_start:layer_end]
            log_probs = F.log_softmax(mask, dim=-1)

            # 对每个 active beam, 生成候选
            candidates = []
            for i, (tok, score, _) in enumerate(active_beams):
                # 如果有 trie 约束, 进一步过滤
                if valid_tokens_per_step is not None and step < len(valid_tokens_per_step):
                    # 获取该 beam 的前缀 (已生成的 SID codes, 不含 BOS)
                    prefix_codes = self.decode_sid(tok[0, 1:])  # [step]
                    prefix_key = tuple(prefix_codes.tolist())

                    # 获取当前前缀允许的下一层 tokens
                    allowed = valid_tokens_per_step[step].get(prefix_key)
                    if allowed is None or len(allowed) == 0:
                        continue  # 死胡同, 跳过

                    # Mask: 只保留允许的 tokens
                    beam_mask = torch.full_like(log_probs[i], float('-inf'))
                    for token_id in allowed:
                        beam_mask[token_id] = log_probs[i, token_id]
                    lp_i = beam_mask
                else:
                    lp_i = log_probs[i]

                top_lp, top_idx = lp_i.topk(min(beam_size, (lp_i > float('-inf')).sum().item()))

                for lp, idx in zip(top_lp, top_idx):
                    if lp.item() == float('-inf'):
                        continue
                    new_tok = torch.cat([tok, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + lp.item()
                    new_done = (idx.item() == self.eos_token_id)
                    candidates.append((new_tok, new_score, new_done))

            candidates.extend(done_beams)

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
            beams = candidates[:beam_size]

        # 提取结果
        sequences = []
        scores = []
        for tok, score, _ in beams:
            gen = tok[0, 1:].tolist()
            gen = [t for t in gen if t != self.eos_token_id]
            if len(gen) < max_len:
                gen += [self.pad_token_id] * (max_len - len(gen))
            sequences.append(gen[:max_len])
            scores.append(score / tok.size(1))

        return (torch.tensor(sequences, device=device),
                torch.tensor(scores, device=device))


# ==============================================================================
# TIGER 数据集
# ==============================================================================

class TIGERDataset(Dataset):
    """
    构建 TIGER 训练的序列数据

    对每个用户, 用滑动窗口创建多个训练样本:
      - 前缀 [i1] → 目标 i2 的 SID
      - 前缀 [i1, i2] → 目标 i3 的 SID
      - ...
      - 前缀 [i1, ..., i_{n-1}] → 目标 i_n 的 SID

    每个输入 token = SID_code[l] + l * K (层偏移)
    """

    def __init__(self, train_df, sid_model, embeddings, item2idx, user2idx,
                 n_embed=256, n_codebook=3, device='cuda'):
        self.n_embed = n_embed
        self.n_codebook = n_codebook

        # ── 为所有物品计算 SID ──
        print('  Computing SIDs for all items...', flush=True)
        sid_model.eval()
        all_items = sorted(embeddings.keys())
        self.item_to_sid = {}

        bs = 512
        with torch.no_grad():
            for start in range(0, len(all_items), bs):
                batch_items = all_items[start:start + bs]
                feats = torch.stack([
                    torch.from_numpy(embeddings[i]).float()
                    for i in batch_items
                ]).to(device)
                codes = sid_model.get_codes(feats)
                codes = codes.squeeze(1).squeeze(1).cpu()
                for item, code in zip(batch_items, codes):
                    self.item_to_sid[item] = code

        # ── 构建用户序列 (向量化加速) ──
        print(f'  Building user sequences...', flush=True)
        import pandas as pd
        user_sequences = {}
        # 使用 groupby 代替逐行迭代
        for user, group in train_df.groupby('user'):
            u = user2idx[user]
            items = []
            for _, row in group.iterrows():
                i = item2idx[row['item']]
                if i in self.item_to_sid:
                    items.append(i)
            if len(items) >= 2:
                user_sequences[u] = items

        # ── 滑动窗口构建样本 ──
        print(f'  Building sliding window samples ({len(user_sequences)} users)...', flush=True)
        self.samples = []
        for u, items in user_sequences.items():
            for t in range(1, len(items)):
                self.samples.append((items[:t], items[t]))

        print(f'  TIGERDataset: {len(self.samples)} 训练样本, '
              f'{len(user_sequences)} 用户, '
              f'{len(self.item_to_sid)} 物品有 SID', flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix_items, target_item = self.samples[idx]

        # 编码源序列: 拼接所有前缀物品的 SID tokens
        src_tokens = []
        for item in prefix_items:
            sid = self.item_to_sid[item]  # [L]
            src_tokens.extend((sid + torch.arange(self.n_codebook) * self.n_embed).tolist())

        # 编码目标: [BOS, sid_c1, sid_c2, sid_c3, EOS]
        target_sid = self.item_to_sid[target_item]
        target_offset = target_sid + torch.arange(self.n_codebook) * self.n_embed
        bos_id = self.n_codebook * self.n_embed + 1  # BOS
        eos_id = self.n_codebook * self.n_embed + 2  # EOS
        tgt_tokens = [bos_id] + target_offset.tolist() + [eos_id]

        return (torch.tensor(src_tokens, dtype=torch.long),
                torch.tensor(tgt_tokens, dtype=torch.long))

    def get_item_to_sid(self):
        return self.item_to_sid


class TIGEREvalDataset:
    """
    TIGER 评估数据: 每个测试用户一条样本
    """

    def __init__(self, test_users, item_to_sid, n_embed=256, n_codebook=3):
        self.n_embed = n_embed
        self.n_codebook = n_codebook
        self.samples = []

        for user_data in test_users:
            target = user_data['target_item']
            train_items = user_data['train_items']

            if target not in item_to_sid:
                continue

            # 源序列: 训练历史物品的 SID
            src_tokens = []
            for item in train_items:
                if item in item_to_sid:
                    sid = item_to_sid[item]
                    src_tokens.extend(
                        (sid + torch.arange(n_codebook) * n_embed).tolist())

            if len(src_tokens) == 0:
                continue

            self.samples.append({
                'user_idx': user_data['user_idx'],
                'src_tokens': torch.tensor(src_tokens, dtype=torch.long),
                'target_item': target,
                'target_sid': item_to_sid[target],
                'train_items': train_items,
            })

        print(f'TIGEREvalDataset: {len(self.samples)} 评估样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_tiger(batch, pad_id=None):
    """
    TIGER batch 整理: 填充变长序列

    Returns:
        src: [B, max_src_len] padded encoder input
        tgt_input: [B, max_tgt_len-1] decoder input (去掉最后一个 EOS)
        tgt_output: [B, max_tgt_len-1] expected output (去掉第一个 BOS)
    """
    if pad_id is None:
        pad_id = 768  # default PAD

    src_list, tgt_list = zip(*batch)

    # Pad src
    max_src = max(s.size(0) for s in src_list)
    src_padded = torch.full((len(src_list), max_src), pad_id, dtype=torch.long)
    for i, s in enumerate(src_list):
        src_padded[i, :s.size(0)] = s

    # Pad tgt
    max_tgt = max(t.size(0) for t in tgt_list)
    tgt_padded = torch.full((len(tgt_list), max_tgt), pad_id, dtype=torch.long)
    for i, t in enumerate(tgt_list):
        tgt_padded[i, :t.size(0)] = t

    # tgt_input = all but last, tgt_output = all but first
    tgt_input = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]

    return src_padded, tgt_input, tgt_output


# ==============================================================================
# TIGER 训练与评估
# ==============================================================================

def train_tiger(model, dataloader, optimizer, device, epochs=200, patience=10):
    """
    训练 TIGER 模型

    Args:
        model: TIGER 模型
        dataloader: TIGER 训练 DataLoader
        optimizer: Adam (lr=3e-4, wd=1e-5)
        device: cuda
        epochs: 最大 epoch 数
        patience: 早停 patience
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(epochs):
        epoch_losses = []

        for src, tgt_in, tgt_out in dataloader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            logits = model(src, tgt_in)  # [B, tgt_len, vocab]

            # Cross-entropy (忽略 PAD)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                tgt_out.reshape(-1),
                ignore_index=model.pad_token_id,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, best={best_loss:.6f}',
                  flush=True)

        if patience_counter >= patience:
            print(f'  Early stop at epoch {epoch+1}', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_tiger(model, eval_dataset, item_to_sid, device,
                   beam_size=100, k_list=(1, 5, 10, 20), report_interval=1000):
    """
    TIGER beam search 评估 (优化版)

    Args:
        model: 训练好的 TIGER 模型
        eval_dataset: TIGEREvalDataset
        item_to_sid: {item: SID_tensor[L]}
        device: cuda
        beam_size: beam search 大小 (default 50 for speed)
        k_list: 评估的 K 值
        report_interval: 每处理多少用户打印进度
    Returns:
        metrics: {HR@K, NDCG@K, ...}
    """
    model.eval()

    # 构建 SID → items 反向索引 + Trie 约束
    sid_to_items = defaultdict(list)
    for item, sid in item_to_sid.items():
        sid_key = tuple(sid.tolist())
        sid_to_items[sid_key].append(item)

    # 构建 Trie: valid_tokens_per_step[step][prefix_tuple] = {allowed_token_ids}
    L = len(next(iter(item_to_sid.values())))  # n_codebook
    K = 256  # n_embed
    valid_tokens_per_step = [
        defaultdict(set),  # step 0: key='()' → {c1 tokens}
        defaultdict(set),  # step 1: key=(c1,) → {c2+256 tokens}
        defaultdict(set),  # step 2: key=(c1,c2) → {c3+512 tokens}
    ]
    for sid_key in sid_to_items.keys():
        c1, c2, c3 = sid_key
        valid_tokens_per_step[0][()].add(c1)           # c1 code
        valid_tokens_per_step[1][(c1,)].add(c2 + K)    # c2 token
        valid_tokens_per_step[2][(c1, c2)].add(c3 + 2*K)  # c3 token

    trie_stats = (
        len(valid_tokens_per_step[0][()]),
        sum(len(v) for v in valid_tokens_per_step[1].values()),
        sum(len(v) for v in valid_tokens_per_step[2].values()),
    )
    print(f'  Trie: {trie_stats[0]} c1, {trie_stats[1]} (c1,c2) pairs, '
          f'{trie_stats[2]} (c1,c2,c3) triples', flush=True)

    metrics = {k: {'HR': [], 'NDCG': []} for k in k_list}
    max_k = max(k_list)
    n_users = len(eval_dataset)

    for idx in tqdm(range(n_users), desc='TIGER beam search'):
        sample = eval_dataset[idx]
        src = sample['src_tokens'].to(device)
        target_item = sample['target_item']

        # Beam search (batched decoder, trie-constrained)
        sequences, _ = model.beam_search(
            src, beam_size=beam_size,
            valid_tokens_per_step=valid_tokens_per_step)
        decoded_sids = model.decode_sid(sequences).cpu()

        # SID → items 映射 (去重)
        seen_items = set()
        ranked_items = []
        for seq in decoded_sids:
            sid_key = tuple(seq.tolist())
            if sid_key in sid_to_items:
                for item in sid_to_items[sid_key]:
                    if item not in seen_items:
                        ranked_items.append(item)
                        seen_items.add(item)
                        if len(ranked_items) >= max_k:
                            break
            if len(ranked_items) >= max_k:
                break

        # 计算指标
        for k in k_list:
            top_k = ranked_items[:k]
            hit = int(target_item in top_k)
            metrics[k]['HR'].append(hit)
            if hit:
                rank = top_k.index(target_item) + 1
                metrics[k]['NDCG'].append(1.0 / np.log2(rank + 1))
            else:
                metrics[k]['NDCG'].append(0.0)

        if (idx + 1) % report_interval == 0:
            hr5 = np.mean(metrics[5]['HR'])
            print(f'  [{idx+1}/{n_users}] HR@5={hr5:.4f}', flush=True)

    results = {}
    for k in k_list:
        results[f'HR@{k}'] = np.mean(metrics[k]['HR'])
        results[f'NDCG@{k}'] = np.mean(metrics[k]['NDCG'])

    return results


# ==============================================================================
# 端到端 TIGER 实验
# ==============================================================================

def run_tiger_evaluation(sid_model, embeddings, train_df, test_df, valid_df,
                         item2idx, user2idx, device,
                         n_codebook=3, n_embed=256,
                         batch_size=256, lr=3e-4, weight_decay=1e-5,
                         epochs=200, patience=10, beam_size=100):
    """
    完整的 TIGER 评估流程:

    1. 用训练好的 SID 模型为所有物品分配 SID
    2. 构建 TIGER 训练序列数据
    3. 训练 TIGER Transformer
    4. Beam search 推理 + 评估

    Args:
        sid_model: 训练好的 SID 模型 (RQ-VAE / QuaSID / ...)
        embeddings: {item_idx: np.ndarray} 文本嵌入
        train_df, test_df, valid_df: 交互数据
        item2idx, user2idx: 映射字典
        device: cuda
    Returns:
        tiget_results: {HR@K, NDCG@K}
        tiger_model: 训练好的 TIGER 模型 (用于后续分析)
    """
    print('\n' + '=' * 60)
    print('TIGER Generative Retrieval Evaluation')
    print('=' * 60)
    print(f'  d_model=128, nhead=8, 4+4 layers, MLP=512, max_seq=80')
    print(f'  lr={lr}, wd={weight_decay}, BS={batch_size}')
    print(f'  epochs={epochs}, beam_size={beam_size}')

    # ── 1. 构建 TIGER 数据 ──
    tiger_ds = TIGERDataset(
        train_df, sid_model, embeddings, item2idx, user2idx,
        n_embed=n_embed, n_codebook=n_codebook, device=device)

    tiger_loader = DataLoader(
        tiger_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn_tiger(
            b, pad_id=n_codebook * n_embed))

    # ── 2. 构建评估数据 ──
    print(f'  Building eval data...', flush=True)
    from data_utils import build_eval_data
    test_users, _ = build_eval_data(
        train_df, test_df, valid_df, embeddings, item2idx, user2idx)
    print(f'  Test users: {len(test_users)}', flush=True)

    eval_ds = TIGEREvalDataset(
        test_users, tiger_ds.get_item_to_sid(),
        n_embed=n_embed, n_codebook=n_codebook)

    # ── 3. 训练 TIGER ──
    print(f'  Training TIGER ({len(tiger_ds)} samples, {len(tiger_loader)} batches/epoch)',
          flush=True)
    tiger_model = TIGER(
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        n_codebook=n_codebook,
        n_embed=n_embed,
    ).to(device)

    total_params = sum(p.numel() for p in tiger_model.parameters())
    print(f'  TIGER params: {total_params:,}')

    optimizer = torch.optim.Adam(
        tiger_model.parameters(), lr=lr, weight_decay=weight_decay)

    tiger_model, history = train_tiger(
        tiger_model, tiger_loader, optimizer, device,
        epochs=epochs, patience=patience)

    # ── 4. Beam search 评估 ──
    # Cleanup to avoid CUDA OOM from fragmentation after long training
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f'\n  Running beam search (beam={beam_size})...')
    results = evaluate_tiger(
        tiger_model, eval_ds, tiger_ds.get_item_to_sid(),
        device, beam_size=beam_size)

    # 打印结果
    print(f'\n  TIGER Results:')
    for k, v in results.items():
        print(f'    {k}: {v:.4f}')

    return results, tiger_model
