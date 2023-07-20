import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from unet_base import VisionTransformer_UNET
from transformer import VisionTransformer
import losses
from val import test_single_volume
import configs

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/vein', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='VEIN/UNET', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mean_teacher', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=5000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[480, 640],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=1000,
                    help='labeled data')
parser.add_argument('--train_num', type=int, default=1000, help='train data number')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

starttime = time.clock()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, config, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    device = torch.device("cuda")

    def create_model(ema=False):
        model = VisionTransformer_UNET(config)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    transformer_refine_model = VisionTransformer(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train_label = BaseDataSets(base_dir=args.root_path, split="train", num=args.train_num, transform=transforms.Compose([
        RandomGenerator(args.patch_size)]))
    db_train_unlabel = BaseDataSets(base_dir=args.root_path, split="train", num=args.train_num, transform=transforms.Compose([
        RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val", num=args.train_num)

    total_slices = len(db_train_label) + len(db_train_unlabel)
    labeled_slice = len(db_train_label)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    trainloader_label = DataLoader(db_train_label, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    trainloader_unlabel = DataLoader(db_train_unlabel, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model = nn.DataParallel(model).cuda()
    model.train()

    ema_model = nn.DataParallel(ema_model).cuda()
    ema_model.train()

    transformer_refine_model = nn.DataParallel(transformer_refine_model).cuda()
    transformer_refine_model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    optimizer_trm = torch.optim.Adam(transformer_refine_model.parameters(), lr=base_lr)
    ce_loss_trm = CrossEntropyLoss()
    dice_loss_trm = losses.DiceLoss(num_classes)
    
    # new top+geo skeleton loss
    skeleton_top_geo = losses.skeleton()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader_label)))
    
    loss_update = 1000

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_label) + 1
    best_performance = 0.0
    best_performance_ema = 0.0
    best_performance_refine = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for (i_batch_l, sampled_batch_l), (i_batch_u, sampled_batch_u) in zip(enumerate(trainloader_label), enumerate(trainloader_unlabel)):
            volume_batch, label_batch = sampled_batch_l['image'], sampled_batch_l['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = sampled_batch_u['image']
            unlabeled_volume_batch = unlabeled_volume_batch.cuda()
            
            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs, _ = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)
            
            outputs_unlabeled, _ = model(unlabeled_volume_batch)
            outputs_soft_unsupervised = torch.sigmoid(outputs_unlabeled)
            
            with torch.no_grad():
                ema_output_l, ema_require_l = ema_model(volume_batch)
                ema_output_soft_l = torch.sigmoid(ema_output_l)
                ema_output_soft_l = torch.argmax(ema_output_soft_l, dim=1).unsqueeze(1).float()

            refine_outputs_l = transformer_refine_model(ema_require_l, ema_output_soft_l)
            refine_outputs_soft_l = torch.sigmoid(refine_outputs_l)

            loss_ce_trm = ce_loss_trm(refine_outputs_l, label_batch[:].long())
            loss_dice_trm = dice_loss_trm(refine_outputs_soft_l, label_batch.unsqueeze(1))
            loss_trm = 0.5 * (loss_dice_trm + loss_ce_trm)
            
            optimizer_trm.zero_grad()
            loss_trm.backward()
            optimizer_trm.step()
            
            with torch.no_grad():
                ema_output, ema_require_u = ema_model(ema_inputs)
                ema_output_soft = torch.sigmoid(ema_output)
                ema_output_soft = torch.argmax(ema_output_soft, dim=1).unsqueeze(1).float()

            refine_outputs = transformer_refine_model(ema_require_u, ema_output_soft)
            refine_outputs_soft = torch.sigmoid(refine_outputs)
            
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            refine_outputs_soft_cldice = torch.argmax(refine_outputs_soft, dim=1).unsqueeze(1).float()
            outputs_soft_cldice = torch.argmax(outputs_soft_unsupervised, dim=1).unsqueeze(1).float()

            loss_skeleton_geo, loss_skeleton_top = skeleton_top_geo(refine_outputs_soft_cldice, outputs_soft_cldice)

            consistency_weight = get_current_consistency_weight(iter_num // args.train_num)

            consistency_loss = torch.mean((outputs_soft_unsupervised - refine_outputs_soft) ** 2)
                    
            loss = supervised_loss + 0.5 * consistency_weight * (consistency_loss + loss_skeleton_geo + loss_skeleton_top)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group_refine in optimizer_trm.param_groups:
                param_group_refine['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('info/total_trm_loss', loss_trm, iter_num)
            writer.add_scalar('info/loss_ce_trm', loss_ce_trm, iter_num)
            writer.add_scalar('info/loss_dice_trm', loss_dice_trm, iter_num)
            writer.add_scalar('info/loss_skeleton_geo', loss_skeleton_geo, iter_num)
            writer.add_scalar('info/loss_skeleton_top', loss_skeleton_top, iter_num)

            logging.info(
                'iteration: %d, loss: %f, loss_ce: %f, loss_dice: %f, loss_cons: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_loss))

            logging.info(
                'total_trm_loss: %f, loss_ce_trm: %f, loss_dice_trm: %f, loss_skeleton_geo: %f, loss_skeleton_top: %f' %
                (loss_trm.item(), loss_ce_trm.item(), loss_dice_trm.item(), loss_skeleton_geo, loss_skeleton_top))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ema_model.eval()
                transformer_refine_model.eval()

                metric_list = 0.0
                metric_list_ema = 0.0
                metric_list_refine = 0.0

                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)

                    if iter_num >= 1000:
                        metric_i_ema = test_single_volume(
                            sampled_batch["image"], sampled_batch["label"], ema_model, classes=num_classes)
                        metric_list_ema += np.array(metric_i_ema)

                    ema_output_test, refine_test = ema_model(sampled_batch["image"].unsqueeze(0).cuda())
                    ema_output_soft_test = torch.sigmoid(ema_output_test)
                    ema_output_soft_test = torch.argmax(ema_output_soft_test, dim=1).unsqueeze(1).float()

                    metric_i_refine = test_single_volume(
                        refine_test, ema_output_soft_test, sampled_batch["label"], transformer_refine_model, classes=num_classes)
                    metric_list_refine += np.array(metric_i_refine)

                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                if iter_num >= 1000:
                    metric_list_ema = metric_list_ema / len(db_val)
                    performance_ema = np.mean(metric_list_ema, axis=0)[0]
                    mean_hd95_ema = np.mean(metric_list_ema, axis=0)[1]
                    logging.info(
                        'iteration %d : mean_dice_ema : %f mean_hd95_ema : %f' % (iter_num, performance_ema, mean_hd95_ema))

                metric_list_refine = metric_list_refine / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice_refine'.format(class_i + 1),
                                      metric_list_refine[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_refine'.format(class_i + 1),
                                      metric_list_refine[class_i, 1], iter_num)

                performance_refine = np.mean(metric_list_refine, axis=0)[0]
                mean_hd95_refine = np.mean(metric_list_refine, axis=0)[1]

                writer.add_scalar('info/val_mean_dice_refine', performance_refine, iter_num)
                writer.add_scalar('info/val_mean_hd95_refine', mean_hd95_refine, iter_num)

                if performance_refine > best_performance_refine:
                    best_performance_refine = performance_refine

                    save_mode_path = os.path.join(snapshot_path,
                                                  'ema_iter_{}_dice.pth'.format(iter_num))
                    save_best = os.path.join(snapshot_path,
                                             'ema_{}_best_model.pth'.format(args.model))
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_best)

                    save_mode_path = os.path.join(snapshot_path,
                                                  'refine_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance_refine, 4)))
                    save_best = os.path.join(snapshot_path,
                                             'refine_{}_best_model.pth'.format(args.model))
                    torch.save(transformer_refine_model.state_dict(), save_mode_path)
                    torch.save(transformer_refine_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice_refine : %f mean_hd95_refine : %f' % (iter_num, performance_refine, mean_hd95_refine))

                model.train()
                ema_model.train()
                transformer_refine_model.train()

            iter_num = iter_num + 1

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    #torch.cuda.set_device(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = configs.get_b16_config()
    train(args, config, snapshot_path)

endtime = time.clock()
print("The train running time is %g s" %(endtime-starttime))