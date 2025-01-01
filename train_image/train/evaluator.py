import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import transforms
from datasets.dataset_brats import BRATS_dataset, RandomGeneratorBRATS
from datasets.dataset_ixi import IXI_dataset, RandomGeneratorIXI
from datasets.dataset_Spine import Spine_dataset, RandomGeneratorSpine
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from utils import EDiceLoss
import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg')


def lpips_metrics(img1, img2):
    img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0)
    img2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0)
    img1 = torch.cat((img1, img1, img1), dim=1)
    img2 = torch.cat((img2, img2, img2), dim=1)
    img1 = 2 * img1 - 1
    img2 = 2 * img2 - 1
    score = loss_fn_vgg(img1, img2)
    return score.item()


def masked_mse(img_o, img_t, seg_mask):
    img_o_masked = img_o[seg_mask == 1]
    img_t_masked = img_t[seg_mask == 1]
    return mse(img_o_masked, img_t_masked)


def masked_mae(img_o, img_t, seg_mask):
    img_o_masked = img_o[seg_mask == 1]
    img_t_masked = img_t[seg_mask == 1]
    return mae(img_o_masked, img_t_masked)


def masked_psnr(img_t, img_o, seg_mask):
    img_o_masked = img_o[seg_mask == 1]
    img_t_masked = img_t[seg_mask == 1]
    return psnr(img_o_masked, img_t_masked)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if np.isfinite(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def split_data(data, inputs, targets):
    contrast_input = inputs
    contrast_output = targets
    img_inputs = [data[i].detach().cuda() for i in contrast_input]
    img_targets = [data[i].detach().cuda() for i in contrast_output]
    return img_inputs, img_targets, contrast_input, contrast_output


def recon_loss(outputs, targets, criterion):
    loss = 0
    for i in range(len(outputs)):
        loss += criterion(outputs[i], targets[i])
    return loss


def eval_brats(model, inputs, targets, dataloader, seg=False, masked=False):
    metrics_meters = [[AverageMeter() for _ in range(5)] for _ in range(len(targets))]
    metrics_meters_masked = [[AverageMeter() for _ in range(3)] for _ in range(len(targets))]
    metrics_meters_seg = [AverageMeter() for _ in range(3)]
    model.eval()
    criterion_seg = EDiceLoss()
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            img_data = data[:-1]
            img_data = [d.detach().cuda() for d in img_data]
            img_inputs, img_targets, contrast_input, contrast_output = split_data(img_data, inputs, targets)
            if seg:
                outputs, _, _ = model(img_inputs, contrast_input, contrast_output + [4])
                img_outputs = outputs[:-1]
                seg_out = outputs[-1]
                seg_gt = data[-1].cuda()
                dices = np.array(criterion_seg.metric(seg_out, seg_gt))
                b = dices.shape[0]
                dices = np.average(dices, axis=0)
                for dice, meter in zip(dices, metrics_meters_seg):
                    meter.update(dice, n=b)
            else:
                img_outputs, _, _ = model(img_inputs, contrast_input, contrast_output)
            for i, outputs in enumerate(img_outputs):
                output_imgs = outputs.detach().cpu().numpy()
                target_imgs = img_targets[i].detach().cpu().numpy()
                for j in range(output_imgs.shape[0]):
                    img_o = output_imgs[j, 0, :, :]
                    img_t = target_imgs[j, 0, :, :]

                    img_max = img_t.max()
                    img_o /= (img_max + 1e-8)  # to avoid nan error
                    img_t /= (img_max + 1e-8)
                    img_o = img_o.clip(0, 1)
                    metrics_meters[i][0].update(mse(img_o, img_t))
                    metrics_meters[i][1].update(mae(img_o, img_t))
                    metrics_meters[i][2].update(ssim(img_o, img_t))
                    metrics_meters[i][3].update(psnr(img_t, img_o))
                    # metrics_meters[i][4].update(lpips_metrics(img_t, img_o))
            if masked:
                for i, outputs in enumerate(img_outputs):
                    output_imgs = outputs.detach().cpu().numpy()
                    target_imgs = img_targets[i].detach().cpu().numpy()
                    seg_masks = data[-1].cpu().numpy()
                    for j in range(output_imgs.shape[0]):
                        seg_mask = seg_masks[j, 2, :, :]
                        if seg_mask.max() > 0:
                            img_o = output_imgs[j, 0, :, :]
                            img_t = target_imgs[j, 0, :, :]
                            img_max = img_t.max()
                            img_o /= img_max
                            img_t /= img_max
                            img_o = img_o.clip(0, 1)
                            metrics_meters_masked[i][0].update(masked_mse(img_o, img_t, seg_mask))
                            metrics_meters_masked[i][1].update(masked_mae(img_o, img_t, seg_mask))
                            metrics_meters_masked[i][2].update(masked_psnr(img_t, img_o, seg_mask))
    metrics_seg = None
    metrics_masked = None
    metrics = [{} for _ in range(len(targets))]
    for i in range(len(targets)):
        metrics[i]['mse'] = metrics_meters[i][0].avg
        metrics[i]['mae'] = metrics_meters[i][1].avg
        metrics[i]['ssim'] = metrics_meters[i][2].avg
        metrics[i]['psnr'] = metrics_meters[i][3].avg
        # metrics[i]['lpips'] = metrics_meters[i][4].avg
    print(f"***Inputs: {inputs}; Outputs: {targets}; {metrics}")
    if seg:
        metrics_seg = {}
        metrics_seg['ET'] = metrics_meters_seg[0].avg
        metrics_seg['TC'] = metrics_meters_seg[1].avg
        metrics_seg['WT'] = metrics_meters_seg[2].avg
        print(f"*Seg: {metrics_seg}")
    if masked:
        metrics_masked = [{} for _ in range(len(targets))]
        for i in range(len(targets)):
            metrics_masked[i]['masked_mse'] = metrics_meters_masked[i][0].avg
            metrics_masked[i]['masked_mae'] = metrics_meters_masked[i][1].avg
            metrics_masked[i]['masked_psnr'] = metrics_meters_masked[i][2].avg
        print(f"***Inputs: {inputs}; Outputs: {targets}; {metrics}")

    return metrics, metrics_seg, metrics_masked


def evaluator_brats(args, model, input_combination, split='test', seg=False, masked=False):
    batch_size = args.batch_size * args.n_gpu
    db = BRATS_dataset(base_dir=args.root_path, split=split,
                       transform=transforms.Compose(
                           [RandomGeneratorBRATS(flip=False, scale=None, n_contrast=4)]))

    print("The length of data set is: {}".format(len(db)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader = DataLoader(db, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False,
                            worker_init_fn=worker_init_fn)

    metrics = []
    metrics_seg = []
    metrics_masked = []
    for inputs in input_combination:
        targets = list(set(range(4)) - set(inputs))
        metric, metric_seg, metric_masked = eval_brats(model, inputs, targets, dataloader, seg=seg, masked=masked)
        metric[0]['input_combination'] = inputs
        metrics.append(metric)
        metrics_seg.append(metric_seg)
        metrics_masked.append(metric_masked)
    return metrics, metrics_seg, metrics_masked


def eval_ixi(model, inputs, targets, dataloader):
    metrics_meters = [[AverageMeter() for _ in range(5)] for _ in range(len(targets))]
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            img_data = [d.detach().cuda() for d in data]
            img_inputs, img_targets, contrast_input, contrast_output = split_data(img_data, inputs, targets)
            img_outputs, _, _ = model(img_inputs, contrast_input, contrast_output)
            for i, outputs in enumerate(img_outputs):
                output_imgs = outputs.detach().cpu().numpy()
                target_imgs = img_targets[i].detach().cpu().numpy()
                for j in range(output_imgs.shape[0]):
                    img_o = output_imgs[j, 0, :, :]
                    img_t = target_imgs[j, 0, :, :]
                    img_max = img_t.max()
                    img_o /= img_max
                    img_t /= img_max
                    img_o = img_o.clip(0, 1)
                    metrics_meters[i][0].update(mse(img_o, img_t))
                    metrics_meters[i][1].update(mae(img_o, img_t))
                    metrics_meters[i][2].update(ssim(img_o, img_t))
                    metrics_meters[i][3].update(psnr(img_t, img_o))
                    metrics_meters[i][4].update(lpips_metrics(img_t, img_o))
    metrics = [{} for _ in range(len(targets))]
    for i in range(len(targets)):
        metrics[i]['mse'] = metrics_meters[i][0].avg
        metrics[i]['mae'] = metrics_meters[i][1].avg
        metrics[i]['ssim'] = metrics_meters[i][2].avg
        metrics[i]['psnr'] = metrics_meters[i][3].avg
        metrics[i]['lpips'] = metrics_meters[i][4].avg
    print(f"***Inputs: {inputs}; Outputs: {targets}; {metrics}")
    return metrics, None, None


def evaluator_ixi(args, model, input_combination, mra_synth=False, split='test'):
    batch_size = args.batch_size * args.n_gpu
    db = IXI_dataset(base_dir=args.root_path, split=split,
                     transform=transforms.Compose(
                         [RandomGeneratorIXI(flip=False, scale=None, n_contrast=4 if args.mra_synth else 3)]))

    print("The length of data set is: {}".format(len(db)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    batch_size = 6  # temporary

    dataloader = DataLoader(db, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    metrics = []
    for inputs in input_combination:
        if args.mra_synth:
            targets = [3]
        else:
            targets = list(set(range(3)) - set(inputs))
        metric, _, _ = eval_ixi(model, inputs, targets, dataloader)
        metrics.append(metric)
    return metrics, None, None


def generate_spine_images(img_data, model, contrast_input, contrast_output, crop_size=256):
    h, w = img_data[0].shape[2:]
    h_n = [i * crop_size for i in range(0, h // crop_size)]
    w_n = [i * crop_size for i in range(0, w // crop_size)]
    if h % crop_size != 0:
        h_n.append(h - crop_size)
    if w % crop_size != 0:
        w_n.append(w - crop_size)
    output_imgs = [torch.zeros_like(img_data[0]) for _ in range(len(contrast_output))]
    for x in h_n:
        for y in w_n:
            img_patches = [img[:, :, x:x + crop_size, y:y + crop_size] for img in img_data]
            output_patches, _, _ = model(img_patches, contrast_input, contrast_output)
            for output_img, output_patch in zip(output_imgs, output_patches):
                output_img[:, :, x:x + crop_size, y:y + crop_size] = output_patch
    return output_imgs


def eval_spine(model, inputs, targets, dataloader):
    metrics_meters = [[AverageMeter() for _ in range(4)] for _ in range(len(targets))]
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            img_data = [d.detach().cuda() for d in data]
            img_inputs, img_targets, contrast_input, contrast_output = split_data(img_data, inputs, targets)
            img_outputs = generate_spine_images(img_inputs, model, contrast_input, contrast_output)
            for i, outputs in enumerate(img_outputs):
                output_imgs = outputs.detach().cpu().numpy()
                target_imgs = img_targets[i].detach().cpu().numpy()
                for j in range(output_imgs.shape[0]):
                    img_o = output_imgs[j, 0, :, :]
                    img_t = target_imgs[j, 0, :, :]
                    img_max = img_t.max()
                    img_o /= img_max
                    img_t /= img_max
                    img_o = img_o.clip(0, 1)
                    metrics_meters[i][0].update(mse(img_o, img_t))
                    metrics_meters[i][1].update(mae(img_o, img_t))
                    metrics_meters[i][2].update(ssim(img_o, img_t))
                    metrics_meters[i][3].update(psnr(img_t, img_o))
    metrics = [{} for _ in range(len(targets))]
    for i in range(len(targets)):
        metrics[i]['mse'] = metrics_meters[i][0].avg
        metrics[i]['mae'] = metrics_meters[i][1].avg
        metrics[i]['ssim'] = metrics_meters[i][2].avg
        metrics[i]['psnr'] = metrics_meters[i][3].avg
    print(f"***Inputs: {inputs}; Outputs: {targets}; {metrics}")
    return metrics


def evaluator_spine(args, model, input_combination, split='test'):
    batch_size = args.batch_size * args.n_gpu
    db = Spine_dataset(base_dir=args.root_path, split=split, random_crop=False,
                       transform=transforms.Compose(
                           [RandomGeneratorSpine(flip=False, scale=None, n_contrast=3)]))

    print("The length of data set is: {}".format(len(db)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader = DataLoader(db, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    metrics = []
    for inputs in input_combination:
        targets = list(set(range(3)) - set(inputs))
        metric = eval_spine(model, inputs, targets, dataloader)
        metrics.append(metric)
    return metrics
