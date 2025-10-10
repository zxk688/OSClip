import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score
from loguru import logger

def compute_entropy(probs):
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
    return entropy

def extract_features(args, model, loader, mode='source'):
    logits_list = []
    labels_list = []
    indices_list = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            if mode == 'source':
                images, labels = data
                images = images.to(args.device)
                labels = labels.to(args.device)
            elif mode == 'target':
                images, indices = data
                images = images.to(args.device)
                labels = None
                indices_list.extend(indices)

            image_features, text_features, logit_scale = model(images)
            logits = logit_scale * torch.matmul(image_features, text_features.T)
            logits_list.append(logits.cpu())
            if labels is not None:
                labels_list.append(labels.cpu())

    if mode == 'source':
        return torch.cat(logits_list), torch.cat(labels_list)
    elif mode == 'target':
        return torch.cat(logits_list), indices_list

def get_pseudo_labels(args, logits, entropy_threshold, energy_threshold):

    probs = F.softmax(logits, dim=-1)
    entropy = compute_entropy(probs)
    energy = -torch.logsumexp(logits, dim=-1)
    
    confidences, pred_labels = torch.max(probs, dim=-1)
    unknown_mask = (energy > energy_threshold) & (entropy > entropy_threshold)
    preds = pred_labels.clone()
    preds[unknown_mask] = args.unknown_class_index

    return preds, confidences, entropy, energy

def compute_weight(confidences):
    min_weight = 0.1
    weights = torch.clamp(confidences, min=min_weight)
    return weights

def pseudo_labeling(args, teacher_model, source_loader, target_loader, entropy_threshold, energy_threshold):
    teacher_model.eval()
    target_logits, target_indices = extract_features(args, teacher_model, target_loader, mode='target')
    target_logits = target_logits.to(args.device)

    pred_labels, confidences, entropy, energy = get_pseudo_labels(args, target_logits, entropy_threshold, energy_threshold)
    weights = compute_weight(confidences).to(args.device)
    return pred_labels, weights

def compute_accuracy(loader, model, args):
    model.eval()
    all_preds = []
    all_labels = []
    all_entropies = []
    all_energy = [] 
    num_total_classes = args.class_num
    class_correct = [0.0 for _ in range(num_total_classes)]
    class_total = [0.0 for _ in range(num_total_classes)]
    class_entropy_sum = [0.0 for _ in range(num_total_classes)]

    with torch.no_grad():
        for data in loader:
            if len(data) == 4:
                images, labels, _, _ = data
            elif len(data) == 2:
                images, labels = data
            else:
                raise ValueError("Invalid dataset mode!")
            
            images = images.to(args.device)
            labels = labels.to(args.device)

            image_features, text_features, logit_scale = model(images)
            logits = logit_scale * torch.matmul(image_features, text_features.T)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)
            entropy = compute_entropy(probs)
            energy = -torch.logsumexp(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_entropies.extend(entropy.cpu().numpy())
            all_energy.extend(energy.cpu().numpy())

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                ent = entropy[i].item()
                class_correct[label] += (pred == label)
                class_total[label] += 1
                class_entropy_sum[label] += ent

    overall_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    per_class_accuracy = {}
    per_class_entropy = {}
    for class_idx in range(args.class_num):
        class_name = args.class_list[class_idx]
        if class_total[class_idx] > 0:
            per_class_accuracy[class_name] = class_correct[class_idx] / class_total[class_idx]
            per_class_entropy[class_name] = class_entropy_sum[class_idx] / class_total[class_idx]
        else:
            per_class_accuracy[class_name] = 0.0
            per_class_entropy[class_name] = 0.0

    labels_list = list(range(args.class_num))
    cm = confusion_matrix(all_labels, all_preds, labels=labels_list)

    TP_per_class = np.diag(cm)
    FP_per_class = np.sum(cm, axis=0) - TP_per_class
    FN_per_class = np.sum(cm, axis=1) - TP_per_class

    num_known_classes = args.class_num - 1  
    KNO = np.mean([TP_per_class[i] / (TP_per_class[i] + FN_per_class[i] + 1e-6) for i in range(num_known_classes)])
    UNK = TP_per_class[-1] / (TP_per_class[-1] + FN_per_class[-1] + 1e-6)
    HOS = 2 * KNO * UNK / (KNO + UNK + 1e-6)
    TP = np.sum(TP_per_class)
    total_samples = np.sum(cm)
    OA = np.mean([TP_per_class[i] / (TP_per_class[i] + FN_per_class[i] + 1e-6) for i in range(args.class_num)])
    IoU_per_class = TP_per_class / (TP_per_class + FP_per_class + FN_per_class + 1e-6)
    mIoU = np.mean(IoU_per_class)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=labels_list)
    average_precisions = []
    for i in range(len(labels_list)):
        class_label = labels_list[i]
        binary_true = [1 if label == class_label else 0 for label in all_labels]
        binary_pred = [1 if pred == class_label else 0 for pred in all_preds]
        if sum(binary_true) == 0:
            average_precision = 0.0
        else:
            average_precision = average_precision_score(binary_true, binary_pred)
        average_precisions.append(average_precision)
    
    macro_average_precision = TP/ total_samples
    logger.info(f"\n{cm}")
    
    return OA, per_class_accuracy, per_class_entropy, macro_average_precision, KNO, UNK, HOS, mIoU, all_energy

def validate(args, model, val_loader):
    overall_acc, per_class_acc, per_class_entropy, macro_avg_prec, KNO, UNK, HOS, mIoU ,all_energy= compute_accuracy(val_loader, model, args)
    return overall_acc, per_class_acc, per_class_entropy, macro_avg_prec, KNO, UNK, HOS, mIoU,all_energy

