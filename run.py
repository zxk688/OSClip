import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
import numpy as np
from loguru import logger
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

import loss
import m_models
import utils as ut
import data.AID_UCMD_NWPU_dataset as dataset_module
from m_argument import get_argument
from torchvision import transforms

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def training_stage_uda(source_loader, target_loader, student_model, teacher_model, 
                       ccl_criterion, wccl_criterion, cross_modal_contrastive_loss_fn,
                       open_set_loss_fn, consistency_loss_fn, optimizer, args, epoch,
                       entropy_threshold, energy_threshold):
    student_model.train()
    total_loss = 0.0
    total_s2text_loss = 0.0
    total_t2text_loss = 0.0
    total_cross_modal_contrastive_loss = 0.0
    total_open_set_loss = 0.0
    total_consistency_loss = 0.0

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for idx in range(max(len(source_loader), len(target_loader))):
        try:
            source_images, source_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_images, source_labels = next(source_iter)

        try:
            target_images, target_indices = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_images, target_indices = next(target_iter)

        source_images = source_images.to(args.device)
        source_labels = source_labels.to(args.device)
        target_images = target_images.to(args.device)

        bcz = source_images.shape[0]

        with torch.no_grad():
            teacher_image_features, teacher_text_features, teacher_logit_scale = teacher_model(target_images)
            teacher_logits = teacher_logit_scale * torch.matmul(teacher_image_features, teacher_text_features.T)
            pred_target_labels, target_confidences, _, _ = ut.get_pseudo_labels(args, teacher_logits, entropy_threshold, energy_threshold)

        target_w = ut.compute_weight(target_confidences).to(args.device)

        images = torch.cat([source_images, target_images], dim=0)
        un_target_pseudo_labels = pred_target_labels.to(args.device)

        image_features, text_features, logit_scale = student_model(images)
        logits = logit_scale * torch.matmul(image_features, text_features.T) / args.temperature

        logits_source = logits[:bcz, :args.class_num-1]
        logits_target = logits[bcz:, :args.class_num]

        s2text_loss = ccl_criterion(logits_source, source_labels)
        t2text_loss = wccl_criterion(logits_target, un_target_pseudo_labels, target_w)
        
        cross_modal_contrastive_loss = cross_modal_contrastive_loss_fn(image_features[:bcz], text_features[:args.class_num-1], source_labels)
        open_set_loss = open_set_loss_fn(logits_target, un_target_pseudo_labels, args)

        consistency_loss = consistency_loss_fn(logits_target, teacher_logits.detach())

        loss_total = (args.lambda_ce * s2text_loss + 
                      args.lambda_wce * t2text_loss +
                      cross_modal_contrastive_loss +
                      args.lambda_OSD * open_set_loss +
                      args.lambda_consistency * consistency_loss)

        if torch.isnan(loss_total):
            logger.error(f"Epoch {epoch}, Batch {idx}: loss NaN")
            raise ValueError("loss NaN")

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_total.item()
        total_s2text_loss += s2text_loss.item()
        total_t2text_loss += t2text_loss.item()
        total_cross_modal_contrastive_loss += cross_modal_contrastive_loss.item()
        total_open_set_loss += open_set_loss.item()
        total_consistency_loss += consistency_loss.item()
            
    max_loader_len = max(len(source_loader), len(target_loader))
    total_loss /= max_loader_len
    total_s2text_loss /= max_loader_len
    total_t2text_loss /= max_loader_len
    total_cross_modal_contrastive_loss /= max_loader_len
    total_open_set_loss /= max_loader_len
    total_consistency_loss /= max_loader_len
    
    return (total_loss,
            total_cross_modal_contrastive_loss, 
            total_s2text_loss, total_t2text_loss, total_open_set_loss,
            total_consistency_loss)
    
def train(args):
    log_dir = os.path.join(args.log_dir, "classification_task")
    ckpt_dir = os.path.join(log_dir, args.dataset_mode, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    now_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_path = os.path.join(log_dir, args.dataset_mode, args.phase + "-" + now_time + ".log")
    logger.add(log_path, rotation='500MB', level="INFO")
    logger.info(args)
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, args.dataset_mode, "tensorboard"))
    
    source_train_dataset, target_train_dataset, val_dataset, test_dataset = dataset_module.get_data(args)
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    student_model = m_models.MYCLIP(args)
    student_model = student_model.to(args.device)

    teacher_model = copy.deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # -------- Phase‑specific prompt handling --------
    if args.training_phase == "phase2":
        student_model.clip_pe.training_phase = "phase2"
        teacher_model.clip_pe.training_phase = "phase2"
        student_model.clip_pe.g_values_target1.requires_grad_(False)

    if args.training_phase == "phase2" and args.load_model:
        student_model.load_state_dict(torch.load(args.load_model))
        teacher_model.load_state_dict(torch.load(args.load_model))
        logger.info(f"Phase2：load_model {args.load_model}")

    ccl_criterion = loss.CentroidsConLoss(args.temperature)
    wccl_criterion = loss.WCentroidsConLoss(args.temperature)
    cross_modal_contrastive_loss_fn = loss.CrossModalContrastiveLoss(temperature=args.temperature)
    open_set_loss_fn = loss.OpenSetDetectionLoss(energy_threshold=-3.0, margin=5.0)
    consistency_loss_fn = loss.ConsistencyLoss(temperature=args.temperature)

    trainable_params = filter(lambda p: p.requires_grad, student_model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    best = 0.0
    current_entropy_threshold = 100
    current_energy_threshold = -3.0  

    for epoch in range(args.max_epoch):
        logger.info(f" Epoch [{epoch+1}/{args.max_epoch}] (train_phase: {args.training_phase})")
        pred_target_labels, target_weights = ut.pseudo_labeling(args, teacher_model, source_train_loader, target_train_loader, current_entropy_threshold, current_energy_threshold)
        
        (train_loss,
         cross_modal_contrastive_loss,
         s2text_loss, t2text_loss, open_set_loss,
         consistency_loss) = training_stage_uda(
            source_train_loader,
            target_train_loader,
            student_model,
            teacher_model,
            ccl_criterion,
            wccl_criterion,
            cross_modal_contrastive_loss_fn,
            open_set_loss_fn,
            consistency_loss_fn,
            optimizer,
            args,
            epoch,
            current_entropy_threshold,
            current_energy_threshold
        )

        overall_acc, per_class_acc, per_class_entropy, macro_avg_prec, KNO, UNK, HOS, mIoU, all_energy = ut.validate(args, student_model, val_loader)
        for class_name, acc in per_class_acc.items():
            logger.info(f"{class_name}acc: {acc:.4f}")

        if len(all_energy) > 0:
            current_energy_threshold = np.percentile(all_energy, 80)
            all_entropies = list(per_class_entropy.values())
            current_entropy_threshold = np.percentile(all_entropies, 80)

        logger.info(f"Epoch [{epoch+1}/{args.max_epoch}], loss: {train_loss:.4f}")
        logger.info(f"KNO: {KNO:.4f}")
        logger.info(f"UNK: {UNK:.4f}")
        logger.info(f"OA: {overall_acc:.4f}")
        logger.info(f"HOS: {HOS:.4f}")
        logger.info(f"mIoU: {mIoU:.4f}")


        update_teacher(student_model, teacher_model, momentum=0.999)

        if HOS > best:
            best = HOS
            model_save_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(student_model.state_dict(), model_save_path)
            logger.info(f"save_best_model HOS: {best:.4f}")

    # --------- Save Gaussian stats for this target domain ---------
    logger.info("Calculate and save the Gaussian distribution statistical parameters of the current target domain …")
    mu, Sigma_inv, logdet = compute_domain_stats(student_model, target_train_loader, args.device)
    stats_path = os.path.join(ckpt_dir, f"stats_{args.training_phase}.pth")
    torch.save({'mu': mu, 'Sigma_inv': Sigma_inv, 'logdet': logdet}, stats_path)
    logger.info(f"Distribution statistics have been saved to {stats_path}")
    logger.info(f"Training completed. Best HOS: {best:.4f}")
    writer.close()

def test(args):
    log_dir = os.path.join(args.log_dir, "classification_task")
    test_log_dir = os.path.join(log_dir, args.dataset_mode, "test_logs")
    os.makedirs(test_log_dir, exist_ok=True)
    
    now_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    test_log_path = os.path.join(test_log_dir, f"test-{now_time}.log")
    logger.add(test_log_path, rotation='500MB', level="INFO")
    logger.info("Start the testing phase")
    logger.info(args)
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(log_dir, args.dataset_mode, "tensorboard"))
    
    student_model = m_models.MYCLIP(args)
    student_model = student_model.to(args.device)
    
    ckpt_dir = os.path.join(log_dir, args.dataset_mode, "checkpoints")
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        logger.error(f"Best model file not found: {best_model_path}")
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")
    student_model.load_state_dict(torch.load(best_model_path))
    logger.info(f"Load best model: {best_model_path}")

    if args.training_phase == "phase2":
        stats1 = torch.load(os.path.join(ckpt_dir, "stats_phase1.pth"), map_location=args.device)
        stats2 = torch.load(os.path.join(ckpt_dir, "stats_phase2.pth"), map_location=args.device)
        student_model.domain_stats = [
            (stats1['mu'].to(args.device), stats1['Sigma_inv'].to(args.device), stats1['logdet']),
            (stats2['mu'].to(args.device), stats2['Sigma_inv'].to(args.device), stats2['logdet'])
        ]
        logger.info("Two-stage Gaussian distribution statistics have been loaded and dynamic fusion inference has been enabled.")
    
    student_model.eval()
    
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=sigma)
    ])
    
    (source_list, target1_train_list, target2_train_list,
     val1_list, val1_list, target1_test_list, target2_test_list) = dataset_module.read_text(args, separate_targets=True)

    if args.training_phase == "phase1":
        target_test_dataset = dataset_module.Test_Dataset(target1_test_list, args.class_num, test_transform)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        overall_acc, per_class_acc, per_class_entropy, macro_avg_prec, KNO, UNK, HOS, mIoU, all_energy = ut.validate(args, student_model, target_test_loader)
        logger.info(f"phase1 test - overall_acc: {overall_acc:.4f}")
        logger.info(f"KNO: {KNO:.4f}")
        logger.info(f"UNK: {UNK:.4f}")
        logger.info(f"OA: {overall_acc:.4f}")
        logger.info(f"HOS: {HOS:.4f}")
        logger.info(f"mIoU: {mIoU:.4f}")
    else:
        target1_test_dataset = dataset_module.Test_Dataset(target1_test_list, args.class_num, test_transform)
        target2_test_dataset = dataset_module.Test_Dataset(target2_test_list, args.class_num, test_transform)
        target1_test_loader = DataLoader(target1_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        target2_test_loader = DataLoader(target2_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        logger.info("Start evaluating the test set of target domain 1")
        overall_acc1, per_class_acc1, per_class_entropy1, macro_avg_prec1, KNO1, UNK1, HOS1, mIoU1, all_energy = ut.validate(args, student_model, target1_test_loader)
        logger.info(f"KNO（: {KNO1:.4f}")
        logger.info(f"UNK: {UNK1:.4f}")
        logger.info(f"OA: {overall_acc1:.4f}")
        logger.info(f"HOS: {HOS1:.4f}")
        logger.info(f"mIoU: {mIoU1:.4f}")
        logger.info("Start evaluating the test set of target domain 2")
        overall_acc2, per_class_acc2, per_class_entropy2, macro_avg_prec2, KNO2, UNK2, HOS2, mIoU2, all_energy = ut.validate(args, student_model, target2_test_loader)
        logger.info(f"KNO（: {KNO2:.4f}")
        logger.info(f"UNK: {UNK2:.4f}")
        logger.info(f"OA: {overall_acc2:.4f}")
        logger.info(f"HOS: {HOS2:.4f}")
        logger.info(f"mIoU: {mIoU2:.4f}")

        logger.info("evaluating the test set of target domain 1 and 2")
        logger.info(f"KNO（: {(KNO1+KNO2)/2:.4f}")
        logger.info(f"UNK: {(UNK1+UNK2)/2:.4f}")
        logger.info(f"OA: {(overall_acc1+overall_acc2)/2:.4f}")
        logger.info(f"HOS: {(HOS1+HOS2)/2:.4f}")
        logger.info(f"mIoU: {(mIoU1+mIoU2)/2:.4f}")
        
        logger.info(f"Domain1 test - overall_acc: {overall_acc1:.4f}")
        logger.info(f"Domain2 test - overall_acc: {overall_acc2:.4f}")
        
    logger.info("Testing phase completed")
    writer.close()
    
def update_teacher(student, teacher, momentum=0.999):
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data

def compute_domain_stats(model, loader, device):
    """
    Compute μ, Σ^{-1}, log|Σ| for a target domain using frozen encoder features.
    Returns (mu, Sigma_inv, logdet)
    """
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats.append(model.frozen_image_features(imgs).cpu())
    feats = torch.cat(feats, dim=0)
    mu = feats.mean(dim=0)
    diff = feats - mu
    Sigma = diff.t() @ diff / feats.size(0) + 1e-6 * torch.eye(diff.size(1))
    Sigma_inv = torch.inverse(Sigma)
    logdet = torch.logdet(Sigma)
    return mu, Sigma_inv, logdet

if __name__ == "__main__":
    args = get_argument()
    if args.phase == "train_uda":
        train(args)
    elif args.phase == "test":
        test(args)