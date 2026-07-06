import torch
import torch.nn.functional as F
import modules.mod_utls as m_utls
from torch.nn.modules.loss import _Loss

Tensor = torch.tensor


def focal_loss(pred, target, gamma=2.0, alpha=0.7, reduction='mean'):
    """
    Focal Loss (二分类, PyTorch版)

    参数:
    pred -- 模型输出的 logits (未过 softmax), [N, 2]
    target -- 真实标签, LongTensor [N]
    gamma -- 调节因子
    reduction -- mean/sum/none
    """
    log_probs = F.log_softmax(pred, dim=-1)  # [N,2]
    probs = torch.exp(log_probs)             # [N,2]

    target = target.view(-1, 1)              # [N,1]
    log_pt = log_probs.gather(1, target).squeeze(1)  # [N]
    pt = probs.gather(1, target).squeeze(1)          # [N]
    # Alpha balancing
    alpha_t = torch.where(target.squeeze() == 1, alpha, 1 - alpha)
    #focal loss公式
    loss = -alpha_t * (1 - pt) ** gamma * log_pt

    if reduction == 'mean':
        loss_value = loss.mean()
    elif reduction == 'sum':
        loss_value = loss.sum()
    else:
        loss_value = loss

    return loss_value, m_utls.to_np(loss_value)


def nll_loss(pred, target, pos_w: float = 1.0):
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor)

    return loss_value, m_utls.to_np(loss_value)


def nll_loss_raw(pred: Tensor, target: Tensor, pos_w,
                 reduction: str = 'mean'):
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor,
                            reduction=reduction)

    return loss_value


class RobustBalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """

    def __init__(self, num_cls, beta):
        super(RobustBalancedSoftmax, self).__init__()
        # num_cls: 类别数量
        self.num_cls = num_cls
        self.beta = beta

    def forward(self, input, label, sample_per_class, reduction='mean', weight=None):
        return robust_balanced_softmax_loss(label, input, sample_per_class, reduction, self.num_cls, self.beta, weight)


def robust_balanced_softmax_loss(labels, logits, sample_per_class, reduction, num_cls, beta, weight=None):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """

    # 将每类样本数转换为与 logits 相同的浮点类型，以便进行数学运算。
    spc = sample_per_class.type_as(logits)
    # 将 spc 从 [C] 扩展到 [BatchSize,C]，使其形状与 logits 匹配
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    # 公式8，调整之后的logits
    # print(logits.shape, spc.shape)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    if weight is not None:
        loss = (weight * loss).mean()

    # 对修改后的 logits 计算 Softmax 概率，作为模型的预测输出 P
    pred = F.softmax(logits, dim=1)
    # 将预测概率 P 限制在 [10的−7,1.0] 范围内
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    # 将整数标签 labels 转换为 One-Hot 编码的真实分布
    label_one_hot = F.one_hot(labels, num_cls).float().to(labels.device)
    # 范围限制
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    # 逆向交叉熵 (RCE) 计算
    rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
    loss += beta * rce.mean()

    return loss


def l2_regularization(model):
    l2_reg = torch.tensor(0., requires_grad=True)
    for key, value in model.named_parameters():
        if len(value.shape) > 1 and 'weight' in key:
            l2_reg = l2_reg + torch.sum(value ** 2) * 0.5
    return l2_reg


