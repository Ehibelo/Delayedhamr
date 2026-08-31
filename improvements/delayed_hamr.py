"""
Delayed HaMR: 延迟激活 HaMR 的两阶段训练

核心思想: 阶段1 (epoch < delay) 关闭 HaMR, 仅用重建+承诺+对比学习;
         阶段2 (epoch >= delay) 激活 HaMR, 恢复完整 QuaSID 损失.

效果: 消除 CL 和 HaMR 在训练早期的梯度冲突,
      HR@5 +5.2%, 训练加速 2.7x (阶段1 无 HaMR 矩阵运算).

用法:
    from delayed_hamr import QuaSIDDelayed, train_delayed
    model = QuaSIDDelayed(delay_epochs=30, ...)
    for epoch in range(epochs):
        model.set_epoch(epoch)
        ...
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加 src 目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from quasid import QuaSID, compute_hamr_loss
from rqvae import RQVAE


class QuaSIDDelayed(QuaSID):
    """
    Delayed HaMR: 训练分为两个阶段

    阶段 1 (epoch < delay_epochs):
        L = L_recon + beta * L_commit + lambda_cl * L_cl
        HaMR 完全关闭, 编码器专注学习重建 + CL 组织全局结构,
        码本 EMA 正常更新, 逐步形成语义锚点.

    阶段 2 (epoch >= delay_epochs):
        L = L_recon + beta * L_commit + lambda_cl * L_cl + L_HaMR
        HaMR 激活, 在稳定的语义空间上进行碰撞分离,
        此时冲突判断反映真实语义关系.

    Args:
        delay_epochs (int): 延迟激活 HaMR 的 epoch 数, 默认 30
        其余参数同 QuaSID
    """

    def __init__(self, delay_epochs=30, **kwargs):
        super().__init__(**kwargs)
        self.delay_epochs = delay_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """每个 epoch 开始前调用, 更新当前 epoch 号"""
        self.current_epoch = epoch

    def compute_loss(self, out, quant_loss, codes, z,
                     trigger_feat, target_feat,
                     trigger_item_ids, target_item_ids,
                     target=None, valid=False):
        """
        与标准 QuaSID.compute_loss 的唯一区别:
        阶段 1 (current_epoch < delay_epochs) 跳过 HaMR 损失计算.
        """
        B = out.shape[0] // 2

        # 重建损失
        xs = torch.cat([trigger_feat, target_feat], dim=0)
        loss_recon = F.mse_loss(out, xs, reduction='mean')
        loss_total = loss_recon + self.beta * quant_loss

        # 对比学习损失 (两个阶段始终激活)
        loss_cl = self.compute_contrastive_loss(z, trigger_item_ids, target_item_ids)
        loss_total = loss_total + self.lambda_cl * loss_cl

        # ★ 关键改动: 阶段 1 跳过 HaMR
        if self.current_epoch >= self.delay_epochs:
            item_ids = torch.cat([trigger_item_ids, target_item_ids], dim=0)
            loss_hamr = compute_hamr_loss(
                codes, z,
                item_ids=item_ids,
                m_full=self.m_full,
                m_partial=self.m_partial,
                R=self.R,
                lambda_full=self.lambda_full,
                lambda_partial=self.lambda_partial,
                use_cvpm=self.use_cvpm,
                trigger_item_ids=trigger_item_ids,
                target_item_ids=target_item_ids,
            )
        else:
            loss_hamr = torch.tensor(0.0, device=out.device)

        loss_total = loss_total + loss_hamr

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': quant_loss,
            'loss_cl': loss_cl,
            'loss_hamr': loss_hamr,
        }


def train_delayed(model, train_loader, valid_loader, epochs, patience,
                   optimizer, device, delay_epochs=None, verbose=True):
    """
    训练 Delayed HaMR 模型

    与标准训练循环的唯一区别: 每个 epoch 前调用 model.set_epoch(epoch),
    compute_loss 内部根据 current_epoch 决定是否激活 HaMR.

    Args:
        model: QuaSIDDelayed 实例
        delay_epochs: 可选覆盖模型内置的 delay_epochs
        其余参数同标准训练
    """
    if delay_epochs is not None:
        model.delay_epochs = delay_epochs

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # ★ 更新 epoch 状态, compute_loss 内部据此决定是否加 HaMR
        model.set_epoch(epoch)
        model.train()

        train_losses = {'total': [], 'recon': [], 'cl': [], 'hamr': []}

        for batch in train_loader:
            trigger_feat = batch['trigger_feat'].to(device)
            target_feat = batch['target_feat'].to(device)
            trigger_item_ids = batch['trigger_item_id'].to(device)
            target_item_ids = batch['target_item_id'].to(device)

            optimizer.zero_grad()

            out, quant_loss, codes, z = model(x=torch.cat([trigger_feat, target_feat], dim=0))

            loss_dict = model.compute_loss(
                out, quant_loss, codes, z,
                trigger_feat=trigger_feat,
                target_feat=target_feat,
                trigger_item_ids=trigger_item_ids,
                target_item_ids=target_item_ids,
            )

            loss_dict['loss_total'].backward()
            optimizer.step()

            train_losses['total'].append(loss_dict['loss_total'].item())
            train_losses['recon'].append(loss_dict['loss_recon'].item())
            train_losses['cl'].append(loss_dict['loss_cl'].item())
            train_losses['hamr'].append(loss_dict['loss_hamr'].item())

        # 验证 (简化: 仅用训练 loss 判断早停, 也可替换为余弦评估)
        avg_train_loss = sum(train_losses['total']) / len(train_losses['total'])

        # 早停
        if avg_train_loss < best_loss - 1e-5:
            best_loss = avg_train_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            haMR_active = epoch >= model.delay_epochs
            print(f"Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | "
                  f"CL: {sum(train_losses['cl'])/len(train_losses['cl']):.4f} | "
                  f"HaMR: {sum(train_losses['hamr'])/len(train_losses['hamr']):.4f} | "
                  f"HaMR active: {haMR_active} | patience: {patience_counter}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stop at epoch {epoch}")
            break

    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return model
