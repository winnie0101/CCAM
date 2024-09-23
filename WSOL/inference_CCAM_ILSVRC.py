import os
import argparse
import time
import torch.backends.cudnn as cudnn
from utils import *
from dataset.ilsvrc import *
from models.loss import *
import random
from models.model import *
from torchvision import transforms
import yaml
from easydict import EasyDict as edict
import torch.distributed as dist
import torch.multiprocessing as mp

# benchmark before running
cudnn.benchmark = True

def parse_arg():
    parser = argparse.ArgumentParser(description="train CCAM on ILSRVC dataset")
    parser.add_argument('--cfg', type=str, default='config/CCAM_ILSVRC.yaml',
                        help='experiment configuration filename')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--port', type=int, default=2345)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--experiment', type=str, default="CCAM_ILSVRC_IP", help='record different experiments')
    parser.add_argument('--pretrained', type=str, default="supervised",
                        help='adopt different pretrained parameters, [supervised, mocov2, detco]')
    parser.add_argument('--evaluate', type=bool, default=False, help='evaluation mode')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)
    config.EXPERIMENT = args.experiment
    config.EVALUTATE = args.evaluate
    config.PORT = args.port
    config.LR = args.lr
    config.ALPHA = args.alpha
    config.EPOCHS = args.epoch
    config.SEED = args.seed
    config.PRETRAINED = args.pretrained

    return config, args

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

    test_data = ILSVRC2012(root=config.ROOT, input_size=480, crop_size=448, train=False, transform=test_transforms)
    print(f'load {len(test_data)} test images!')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.BATCH_SIZE,
        num_workers=config.WORKERS, pin_memory=True, collate_fn=my_collate, sampler=test_sampler)

    criterion = [SimMaxLoss(metric='cos', alpha=config.ALPHA).cuda(local_rank), 
                 SimMinLoss(metric='cos').cuda(local_rank),
                 SimMaxLoss(metric='cos', alpha=config.ALPHA).cuda(local_rank)]

    best_CorLoc, best_threshold = run_inference(config, test_loader, model, criterion)
    print('Best CorLoc: {:.2f}, Best Threshold: {}'.format(best_CorLoc, best_threshold))
    
    
    run_inference(config, test_loader, model, criterion, is_extract=True, threshold=best_threshold)


if __name__ == "__main__":
    main()