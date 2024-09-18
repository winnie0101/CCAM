import torch
import time
import yaml
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms
from dataset.cub200 import CUB200
from models.model import get_model
from easydict import EasyDict as edict
from utils import *
from models.loss import SimMaxLoss, SimMinLoss
import os

cudnn.benchmark = True
os.environ["NUMEXPR_NUM_THREADS"] = "16"

def parse_arg():
    parser = argparse.ArgumentParser(description="train CCAM on CUB dataset")
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='config/CCAM_CUB.yaml')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--experiment', type=str, default="CCAM_CUB_IP", help='record different experiments')
    parser.add_argument('--pretrained', type=str, default="supervised", help='adopt different pretrained parameters, [supervised, mocov2, detco]')
    
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    config.EXPERIMENT = args.experiment
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.PRETRAINED = args.pretrained

    return config, args

def process_ccam(ccam):
    flag = check_positive(ccam.clone())
    if flag:
        ccam = 1 - ccam
    return ccam

def run_inference(config, data_loader, model, criterion, is_extract=False, threshold=None):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    threshold_list = [(i + 1) / config.NUM_THRESHOLD for i in range(config.NUM_THRESHOLD - 1)] if not is_extract else [threshold]
    Corcorrect = torch.Tensor([[0] for _ in range(len(threshold_list))]) if not is_extract else None
    total = 0
    end = time.time()

    with torch.no_grad():
        for i, data in enumerate(data_loader, start=0):
            input, target, bboxes, cls_name, img_name = data
            input = input.cuda()

            fg_feats, bg_feats, ccam = model(input)
            ccam = process_ccam(ccam)

            pred_boxes_t = generate_pred_boxes(input, ccam, threshold_list)

            if not is_extract:
                calculate_loss(fg_feats, bg_feats, criterion, losses, input)

                Corcorrect, total = compute_corloc(Corcorrect, pred_boxes_t, bboxes, threshold_list, total)

                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    print(f'Test: [{i}/{len(data_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss {losses.val:.4f} ({losses.avg:.4f})')

            else:
                save_bbox_as_json(config, config.EXPERIMENT, i, 0, pred_boxes_t[0], cls_name, img_name, phase='test')
                if i % config.PRINT_FREQ == 0:
                    print(f'Extracted [{i}/{len(data_loader)}]')
                    visualize_heatmap(config, config.EXPERIMENT, input.clone().detach(), ccam, cls_name, img_name, phase='test', bboxes=pred_boxes_t[0], gt_bboxes=bboxes)

    if not is_extract:
        current_best_CorLoc = max([(Corcorrect[i].item() / total) * 100 for i in range(len(threshold_list))])
        return current_best_CorLoc, threshold_list[Corcorrect.argmax().item()]
    
    return None, None

def generate_pred_boxes(input, ccam, threshold_list):
    pred_boxes_t = [[] for _ in range(len(threshold_list))]
    for j in range(input.size(0)):
        estimated_boxes_at_each_thr, _ = compute_bboxes_from_scoremaps(
            ccam[j, 0, :, :].detach().cpu().numpy().astype(np.float32),
            threshold_list, input.size(-1) / ccam.size(-1), multi_contour_eval=False)
        for k in range(len(threshold_list)):
            pred_boxes_t[k].append(estimated_boxes_at_each_thr[k])
    return pred_boxes_t

def calculate_loss(fg_feats, bg_feats, criterion, losses, input):
    loss1 = criterion[0](bg_feats)
    loss2 = criterion[1](bg_feats, fg_feats)
    loss3 = criterion[2](fg_feats)
    loss = loss1 + loss2 + loss3
    losses.update(loss.data.item(), input.size(0))

def compute_corloc(Corcorrect, pred_boxes_t, bboxes, threshold_list, total):
    total += len(bboxes)
    for j in range(len(threshold_list)):
        pred_boxes = pred_boxes_t[j]
        pred_boxes = torch.from_numpy(np.array([pred_boxes[k][0] for k in range(len(pred_boxes))])).float()
        gt_boxes = bboxes[:, 1:].float()

        inter = intersect(pred_boxes, gt_boxes)
        area_a = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_b = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        union = area_a + area_b - inter
        IOU = inter / union
        IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
        IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))
        Corcorrect[j] += IOU.sum()

    return Corcorrect, total

def main():
    config, args = parse_arg()
    model_path = f'./debug/checkpoints/{config.EXPERIMENT}/current_epoch.pth'

    print("=> Loading model...")
    model = get_model(pretrained=config.PRETRAINED).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize(size=(480, 480)),
        transforms.CenterCrop(size=(448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_data = CUB200(root=config.ROOT, input_size=480, crop_size=448, train=False, transform=test_transforms)
    print(f'load {len(test_data)} test images!')
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.WORKERS, pin_memory=True)

    criterion = [SimMaxLoss(metric='cos', alpha=args.alpha).cuda(),
                 SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=args.alpha).cuda()]

    best_CorLoc, best_threshold = run_inference(config, test_loader, model, criterion)
    print('Best CorLoc: {:.2f}, Best Threshold: {}'.format(best_CorLoc, best_threshold))
    
    
    run_inference(config, test_loader, model, criterion, is_extract=True, threshold=best_threshold)

if __name__ == "__main__":
    main()
