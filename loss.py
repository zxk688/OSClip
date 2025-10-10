import torch
import torch.nn as nn
import torch.nn.functional as F

class CentroidsConLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(CentroidsConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        return loss

class WCentroidsConLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(WCentroidsConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits, labels, weights):
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss * weights).mean()
        return loss

class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features, labels):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        loss = self.cross_entropy(logits, labels)
        return loss

class PrototypicalConstraintLoss(nn.Module):
    def __init__(self):
        super(PrototypicalConstraintLoss, self).__init__()

    def forward(self, features, source_prototypes):
        distances = torch.cdist(features, source_prototypes)
        exp_neg_dist = torch.exp(-distances)
        softmax_dist = exp_neg_dist / torch.sum(exp_neg_dist, dim=1, keepdim=True)
        min_probs, _ = torch.min(softmax_dist, dim=1)
        loss = torch.mean(min_probs)
        return loss

class OpenSetDetectionLoss(nn.Module):
    def __init__(self, energy_threshold, margin=3.0):
        super(OpenSetDetectionLoss, self).__init__()
        self.energy_threshold = energy_threshold
        self.margin = margin

    def forward(self, logits, labels, args):
        energy = -torch.logsumexp(logits, dim=-1)
        known_mask = (labels != args.unknown_class_index).float()
        unknown_mask = 1.0 - known_mask
        
        loss_known = known_mask * F.relu(energy - self.energy_threshold + self.margin)
        loss_unknown = unknown_mask * F.relu(self.energy_threshold - energy + self.margin)
        loss = (loss_known.sum() + loss_unknown.sum()) / logits.size(0)
        return loss

class ConsistencyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        return loss