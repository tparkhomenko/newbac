import torch
import torch.nn as nn
import torch.nn.functional as F


class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, alpha=0.5, beta=0.5, gamma=2.0):
        super().__init__()
        # LDAM margins (1/sqrt(class_count))
        m_list = 1.0 / torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float))
        self.margins = m_list / m_list.max()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Simple focal loss implementation (since we don't have external focal loss)
        self.focal_gamma = gamma

    def forward(self, logits, targets):
        # LDAM component
        margins = self.margins.to(logits.device)
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        batch_margins = torch.matmul(index.float(), margins.unsqueeze(1)).view(-1)
        logits_m = logits - batch_margins.unsqueeze(1)
        ldam_loss = F.cross_entropy(logits_m, targets, reduction='mean')

        # Focal component
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        focal_loss = focal_loss.mean()

        # Combine
        return self.alpha * ldam_loss + self.beta * focal_loss
