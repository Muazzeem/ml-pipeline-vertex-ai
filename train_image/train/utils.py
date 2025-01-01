import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
from torch.nn import functional as F
import SimpleITK as sitk
from torchvision.utils import make_grid
from torchvision import models, transforms


def make_image_grid(tensor_list):
    images = []
    for tensor in tensor_list:
        image = tensor[0, 0, :, :].unsqueeze(0)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.repeat(3, 1, 1)
        image = torch.rot90(image, 3, [1, 2])
        images.append(image)
    image_grid = make_grid(images, padding=0)
    return image_grid


def make_seg_grid(tensor):
    images = []
    n_channel = tensor.shape[1]
    for i in range(n_channel):
        image = tensor[0, i, :, :].unsqueeze(0)
        image = image.repeat(3, 1, 1)
        image = torch.rot90(image, 3, [1, 2])
        images.append(image)
    image_grid = make_grid(images, padding=0)
    return image_grid


def list2str(x):
    f = ''
    for x_i in x:
        f += f'{x_i}'
    return f


class VGGLoss(nn.Module):
    def __init__(self, resize=True, requires_grad=False, lossfn_type='mse'):
        super(VGGLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg19(pretrained=True).features[:3].eval())
        blocks.append(models.vgg19(pretrained=True).features[3:8].eval())
        blocks.append(models.vgg19(pretrained=True).features[8:17].eval())
        # blocks.append(models.vgg19(pretrained=True).features[17:26].eval())
        # blocks.append(models.vgg19(pretrained=True).features[26:35].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = requires_grad
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize

        # RGB mean, std
        # https://pytorch.org/vision/stable/models.html
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        if lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()

    def vgg_feat(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std

        if self.resize:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)

        feat = []
        feat.append(x)
        for block in self.blocks:
            x = block(x).div(12.75)
            feat.append(x)

        return feat

    def forward(self, x, gt):
        x_vgg, gt_vgg = self.vgg_feat(x), self.vgg_feat(gt)
        loss = 0.
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += (1. / n) * self.lossfn(x_vgg[i], gt_vgg[i].detach())
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True).item())
            dices.append(dice)
        return dices


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
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
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list
