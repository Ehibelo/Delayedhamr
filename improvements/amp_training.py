"""
AMP 混合精度训练 — 加速 QuaSID
===============================

动机:
  算法层面改动（稀疏HaMR、去CL等）要么不加速、要么精度崩溃。
  工程层面: AMP (Automatic Mixed Precision) 将矩阵运算从 float32 → float16,
  RTX 4090 的 Tensor Core 对 fp16 有硬件加速，理论加速 1.5-2×。

方法:
  - torch.cuda.amp.autocast() 包裹前向传播
  - GradScaler 处理 fp16 梯度溢出
  - 不改变任何模型结构或损失函数

用法:
  from improvements.amp_training import train_quasid_amp
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from quasid import QuaSID


def train_quasid_amp(model, pair_loader, device, epochs=200, patience=10,
                     lr=3e-4, weight_decay=1e-5, log_interval=10,
                     use_amp=True):
    """AMP 混合精度训练 QuaSID

    Args:
        use_amp: True = 开启 AMP, False = 标准 fp32 (对照)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = defaultdict(list)

    for epoch in range(epochs):
        epoch_losses = defaultdict(list)

        for trigger_feat, target_feat, trigger_id, target_id in pair_loader:
            trigger_feat = trigger_feat.to(device)
            target_feat = target_feat.to(device)
            trigger_id = trigger_id.to(device)
            target_id = target_id.to(device)

            optimizer.zero_grad()

            # ★ AMP: autocast 包裹前向传播
            with torch.cuda.amp.autocast(enabled=use_amp):
                feats = torch.cat([trigger_feat, target_feat], dim=0)
                out, quant_loss, codes, z, z_q = model(feats)

                loss_dict = model.compute_loss(
                    out, quant_loss, codes, z,
                    trigger_feat, target_feat, trigger_id, target_id,
                )

            # ★ AMP: scaler 处理梯度
            scaler.scale(loss_dict['loss_total']).backward()
            scaler.step(optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())

        avg_loss = np.mean(epoch_losses['loss_total'])
        for k in epoch_losses:
            history[k].append(np.mean(epoch_losses[k]))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0:
            print(f'  Epoch {epoch+1:3d}: loss={avg_loss:.6f}  '
                  f'patience={patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'  早停 @ epoch {epoch+1}')
            break

    model.load_state_dict(best_state)
    return model, dict(history)
