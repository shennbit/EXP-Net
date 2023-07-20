import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils.soft_skeleton import soft_skel
from PIL import Image
from skimage.io import imsave
import os
from skimage import morphology


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class Soft_cldice(nn.Module):
    def __init__(self, iter_=25, smooth=1e-5):
        super(Soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, target, inputs):
        with torch.no_grad():
            skel_pred = soft_skel(inputs, self.iter)
            skel_true = soft_skel(target, self.iter)
            
        #if iter_num==400:
        #    skel_pred_1 = skel_pred.cpu().detach().numpy()
        #    pred_dir = 'preds0105'
        #    skel_pred_1 = (skel_pred_1[0,0,:,:] * 255.).astype(np.uint8)
        #    imsave(os.path.join(pred_dir, "".join("0p.jpg")), skel_pred_1)
            
        #    skel_true_1 = skel_true.cpu().detach().numpy()
        #    skel_true_1 = (skel_true_1[0,0,:,:] * 255.).astype(np.uint8)
        #    imsave(os.path.join(pred_dir, "".join("1l.jpg")), skel_true_1)

        y_pred = inputs
        y_true = target

        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
                    torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
                    torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

        return cl_dice
        
        
class skeleton(nn.Module):
    def __init__(self, smooth=1e-5):
        super(skeleton_cldice, self).__init__()
        self.smooth = smooth

    def forward(self, target, inputs): #target: label, inputs: prediction
        y_true = target
        y_pred = inputs
        
        with torch.no_grad():
            target_numpy = target.cpu().numpy()
            inputs_numpy = inputs.cpu().numpy()
            
            target_tensor_geo = []
            target_tensor_top = []
            
            inputs_tensor_geo = []
            inputs_tensor_top = []
            
            for i in range(target_numpy.shape[0]):
                #target
                target_numpy_i = target_numpy[i,0,:,:]
                
                ske_geo_target_numpy_i = morphology.medial_axis(target_numpy_i)
                ske_top_target_numpy_i = morphology.thin(target_numpy_i)
                
                target_tensor_geo.append(ske_geo_target_numpy_i)
                target_tensor_top.append(ske_top_target_numpy_i)
                
                #inputs
                inputs_numpy_i = inputs_numpy[i,0,:,:]
                
                ske_geo_inputs_numpy_i = morphology.medial_axis(inputs_numpy_i)
                ske_top_inputs_numpy_i = morphology.thin(inputs_numpy_i)
                
                inputs_tensor_geo.append(ske_geo_inputs_numpy_i)
                inputs_tensor_top.append(ske_top_inputs_numpy_i)
                
            #target
            target_tensor_geo = np.array(target_tensor_geo)
            target_tensor_top = np.array(target_tensor_top)
            
            target_tensor_geo = torch.tensor(target_tensor_geo)
            target_tensor_geo = target_tensor_geo.cuda()
            
            target_tensor_top = torch.tensor(target_tensor_top)
            target_tensor_top = target_tensor_top.cuda()
            
            target_tensor_geo = target_tensor_geo.unsqueeze(1).float()
            target_tensor_top = target_tensor_top.unsqueeze(1).float()
            
            #inputs
            inputs_tensor_geo = np.array(inputs_tensor_geo)
            inputs_tensor_top = np.array(inputs_tensor_top)
            
            inputs_tensor_geo = torch.tensor(inputs_tensor_geo)
            inputs_tensor_geo = inputs_tensor_geo.cuda()
            
            inputs_tensor_top = torch.tensor(inputs_tensor_top)
            inputs_tensor_top = inputs_tensor_top.cuda()
            
            inputs_tensor_geo = inputs_tensor_geo.unsqueeze(1).float()
            inputs_tensor_top = inputs_tensor_top.unsqueeze(1).float()
            

        #geo
        skel_pred_geo = inputs_tensor_geo
        skel_true_geo = target_tensor_geo
        
        tprec_geo = (torch.sum(torch.multiply(skel_pred_geo, y_true)) + self.smooth) / (
                    torch.sum(skel_pred_geo) + self.smooth)
        tsens_geo = (torch.sum(torch.multiply(skel_true_geo, y_pred)) + self.smooth) / (
                    torch.sum(skel_true_geo) + self.smooth)
        cl_dice_geo = 1. - 2.0 * (tprec_geo * tsens_geo) / (tprec_geo + tsens_geo)
        
        #top
        skel_pred_top = inputs_tensor_top
        skel_true_top = target_tensor_top
        
        tprec_top = (torch.sum(torch.multiply(skel_pred_top, y_true)) + self.smooth) / (
                    torch.sum(skel_pred_top) + self.smooth)
        tsens_top = (torch.sum(torch.multiply(skel_true_top, y_pred)) + self.smooth) / (
                    torch.sum(skel_true_top) + self.smooth)
        cl_dice_top = 1. - 2.0 * (tprec_top * tsens_top) / (tprec_top + tsens_top)

        return cl_dice_geo, cl_dice_top


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map
