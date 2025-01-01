import argparse
import logging
import os
import pdb
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torchvision import transforms
from datasets.dataset_brats import BRATS_dataset, RandomGeneratorBRATS
from evaluator import evaluator_brats, AverageMeter
import pdb
from timm.scheduler.cosine_lr import CosineLRScheduler

from save_model_on_gcs import store_model
from save_latest_model import copy_model
from utils import list2str, make_image_grid, EDiceLoss, make_seg_grid, VGGLoss
from loss_ssim import MSF_SSIM


input_combination = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2],
                     [0, 1, 3], [0, 2, 3], [1, 2, 3]]

input_combination_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]



def cosine_similarity(x, y):
    cos = nn.CosineSimilarity(dim=0)
    return cos(x.view(-1), y.view(-1))


# compute triplet loss for each sample
def compute_triplet_loss(sample_1, sample_2, margin=0.1):
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_similarity, margin=margin)
    n_contrast = sample_1.shape[0]
    loss = 0
    for i in range(n_contrast):
        for j in range(n_contrast):
                anchor1 = sample_1[i]
                positive1 = sample_1[j]
                negative1 = sample_2[j]
                anchor2 = sample_2[i]
                positive2 = sample_2[j]
                negative2 = sample_1[j]
                loss += triplet_loss(anchor1, positive1, negative1) + triplet_loss(anchor2, positive2, negative2)
    loss = loss/(2*n_contrast**2)
    return loss


def contrastive_loss(enc_outs, idx_i, idx_j, margin=0.1):
    loss = 0
    for enc_out in enc_outs:
        # enc_out: B n_contrast H W C
        sample_i = enc_out[idx_i, :, :, :, :]
        sample_j = enc_out[idx_j, :, :, :, :]
        loss += compute_triplet_loss(sample_i, sample_j, margin)
    loss = loss/len(enc_outs)
    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def random_missing_input(n, k=None):
    contrasts = list(range(n))
    random.shuffle(contrasts)
    if not k:
        k = np.random.randint(1, n)   #k: number of inputs in [1, n_contrast)
    contrast_input = sorted(contrasts[:k])
    contrast_output = sorted(contrasts[k:])
    return contrast_input, contrast_output


def random_split_data(data, k=None, zero_gad=False):
    n_contrast = len(data)
    contrasts = list(range(n_contrast))
    if zero_gad:
        contrast_input = [0, 2, 3]
        contrast_output = [1]
    else:
        random.shuffle(contrasts)
        if not k:
            k = np.random.randint(1, n_contrast) #k: number of inputs in [1, n_contrast)
        contrast_input = sorted(contrasts[:k])
        contrast_output = sorted(contrasts[k:])
    img_inputs = [data[i]  for i in contrast_input]
    img_targets = [data[i] for i in contrast_output]
    return img_inputs, img_targets, contrast_input, contrast_output

def all_splits(data, k=None):
    n_contrast = len(data)
    contrasts = list(range(n_contrast))

    return_sets = []

    for c in contrasts:
        contrast_output = [c]
        contrast_input = sorted([con for con in contrasts if con != c])

        img_inputs = [data[i]  for i in contrast_input]
        img_targets = [data[i] for i in contrast_output]

        return_sets.append((img_inputs, img_targets, contrast_input, contrast_output))

    return return_sets

def recon_loss(outputs, targets, criterion, enh_weight=False, enh_weight_input=None):
    if enh_weight:
        img_ref = enh_weight_input[0] # for t1 pre
        img_con = targets[0] # for t1 post
        enh_mask = (img_con - img_ref) + 1e-8
    loss = 0
    for output, target in zip(outputs, targets):
        if enh_weight:
            output_weighted = output * enh_mask
            target_weighted = target * enh_mask
            loss += criterion(output_weighted, target_weighted)
        else:
            loss += criterion(output, target)
    loss = loss/len(outputs)
    return loss

def trainer_brats(args, G, D, snapshot_path, model_dir):
    best_performance_mae = np.inf
    best_performance_mse = np.inf
    best_performance_psnr = 0
    best_performance_ssim = 0

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr_g = args.base_lr_g
    base_lr_d = args.base_lr_d
    batch_size = args.batch_size * args.n_gpu
    db_train = BRATS_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGeneratorBRATS(flip=True, scale=[0.9, 1.1])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D) if args.lambda_GAN > 0 else None
    if args.optimizer == 'sgd':
        opt_G = optim.SGD(G.parameters(), lr=base_lr_g, momentum=0.9, weight_decay=0.0001)
        opt_D = optim.SGD(D.parameters(), lr=base_lr_d, momentum=0.9, weight_decay=0.0001) if args.lambda_GAN > 0 else None
    elif args.optimizer == 'adamw':
        opt_G = optim.AdamW(G.parameters(), lr=base_lr_g)
        opt_D = optim.AdamW(D.parameters(), lr=base_lr_d) if args.lambda_GAN > 0 else None
    else:
        opt_G = optim.Adam(G.parameters(), lr=base_lr_g, weight_decay=0.00001)
        opt_D = optim.Adam(D.parameters(), lr=base_lr_d, weight_decay=0.00001) if args.lambda_GAN > 0 else None

    lr_scheduler_g = CosineLRScheduler(
        opt_G,
        t_initial=args.max_epochs * len(trainloader),
        t_mul=1.,
        lr_min=5e-6,
        warmup_lr_init=5e-7,
        warmup_t=args.warmup_epoch * len(trainloader),
        cycle_limit=1,
        t_in_epochs=False
    )
    lr_scheduler_d = CosineLRScheduler(
        opt_D,
        t_initial=args.max_epochs * len(trainloader),
        t_mul=1.,
        lr_min=1e-6,
        warmup_lr_init=1e-7,
        warmup_t=args.warmup_epoch * len(trainloader),
        cycle_limit=1,
        t_in_epochs=False
    ) if args.lambda_GAN > 0 else None

    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location='cpu')
        G.load_state_dict(state_dict['G'])
        if args.lambda_GAN > 0:
            D.load_state_dict(state_dict['D'])

        if args.continue_training:
            args.start_epoch = state_dict['epoch'] + 1

        best_performance_mae = state_dict['mae']
        best_performance_mse = state_dict['mse']
        if 'psnr' in state_dict:
            best_performance_psnr = state_dict['psnr']
        else:
            best_performance_psnr = 0
        if 'ssim' in state_dict:
            best_performance_ssim = state_dict['ssim']
        else:
            best_performance_ssim = 0

        opt_G.load_state_dict(state_dict['opt_G'])
        if args.lambda_GAN > 0:
            opt_D.load_state_dict(state_dict['opt_D'])
        lr_scheduler_g.load_state_dict(state_dict['lr_scheduler_g'])
        if args.lambda_GAN > 0:
            lr_scheduler_d.load_state_dict(state_dict['lr_scheduler_d'])

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    model_G = G.module if args.n_gpu>1 else G
    if args.lambda_GAN > 0:
        model_D = D.module if args.n_gpu>1 else D

    criterion_img = L1Loss()
    criterion_ssim = MSF_SSIM(
        img_shape=[1, model_G.img_size[0], model_G.img_size[1]], data_range=10., size_average=True, kadid10k_opt=True
    ).to('cuda')
    criterion_seg = EDiceLoss(do_sigmoid=True)
    criterion_perceptual = VGGLoss().to('cuda')

    for epoch_num in trange(args.start_epoch, max_epoch):
        G.train()
        if args.lambda_GAN > 0:
            D.train()
        loss_meter_g = AverageMeter()
        if args.lambda_GAN > 0:
            loss_meter_gan_g = AverageMeter()
            loss_meter_gan_d = AverageMeter()
        if args.lambda_seg > 0:
            loss_meter_seg = AverageMeter()
        if args.lambda_perceptual > 0:
            loss_meter_perceptual = AverageMeter()
        if args.lambda_ssim > 0:
            loss_meter_ssim = AverageMeter()

        loss_meter_rec = AverageMeter()
        n_contrast = model_G.n_contrast
        contrasts = list(range(n_contrast))
        for i_batch, data in enumerate(tqdm(trainloader)):
            # import pdb
            # pdb.set_trace()
            img_data = data[:-1]
            img_data = [d.detach().cuda() for d in img_data]

            # img_inputs, img_targets, contrast_inputs, contrast_outputs = random_split_data(img_data, args.k, zero_gad=args.zero_gad)

            return_sets = all_splits(img_data, args.k)

            for (img_inputs, img_targets, contrast_inputs, contrast_outputs) in return_sets:

                iter_num = iter_num + 1
                loss_g = 0

                if args.lambda_self > 0 and args.lambda_seg > 0:
                    outputs, enc_out, _ = G(img_inputs, contrast_inputs, contrasts+[4])
                    # compute image loss
                    img_outs = outputs[:-1]
                    img_inputs_recon = [img_outs[contrast_i] for contrast_i in contrast_inputs]
                    img_outputs = [img_outs[contrast_o] for contrast_o in contrast_outputs]
                    loss_input_rec = recon_loss(img_inputs_recon, img_inputs, criterion_img)
                    loss_g += args.lambda_self * loss_input_rec
                    writer.add_scalar('train/loss_input_rec', loss_input_rec, iter_num)
                    # compute seg loss
                    seg_out = outputs[-1]
                    seg_gt = data[-1].cuda()
    #                 if torch.sum(seg_gt) > 50:
    #                     writer.add_image('train/seg_gt', make_seg_grid(seg_gt), iter_num)
    #                     pdb.set_trace()
    #                     from torchvision.utils import save_image
    #                     save_image(make_seg_grid(seg_gt), "test.png")
    #                     pdb.set_trace()
                    loss_seg = criterion_seg(seg_out, seg_gt)
                    loss_g += args.lambda_seg * loss_seg
                    writer.add_scalar('train/loss_seg', loss_seg, iter_num)
                    loss_meter_seg.update(loss_seg.item())

                elif args.lambda_self > 0:
                    # compute self reconstruction loss
                    img_outs, enc_out, _ = G(img_inputs, contrast_inputs, contrasts)
                    img_inputs_recon = [img_outs[contrast_i] for contrast_i in contrast_inputs]
                    img_outputs = [img_outs[contrast_o] for contrast_o in contrast_outputs]
                    loss_input_rec = recon_loss(img_inputs_recon, img_inputs, criterion_img)
                    loss_g += args.lambda_self*loss_input_rec
                    writer.add_scalar('train/loss_input_rec', loss_input_rec, iter_num)
                elif args.lambda_seg > 0:
                    outputs, enc_out, _ = G(img_inputs, contrast_inputs, contrast_outputs+[4])
                    # compute image loss
                    img_outputs = outputs[:-1]
                    # compute seg loss
                    seg_out = outputs[-1]
                    seg_gt = data[-1].cuda()
                    loss_seg = criterion_seg(seg_out, seg_gt)
                    loss_g += args.lambda_seg * loss_seg
                    writer.add_scalar('train/loss_seg', loss_seg, iter_num)
                    loss_meter_seg.update(loss_seg.item())
                else:
                    img_outputs, enc_out, _ = G(img_inputs, contrast_inputs, contrast_outputs)

                # compute cross recontruction loss
                loss_output_rec = recon_loss(
                    img_outputs, img_targets, criterion_img, enh_weight=args.enh_weight,
                    enh_weight_input=img_inputs if args.enh_weight else None
                )
                loss_g += args.lambda_cross * loss_output_rec
                writer.add_scalar('train/loss_output_rec', loss_output_rec, iter_num)

                # compute triplet loss
                if args.lambda_triplet > 0:
                    # randomly pick two samples each batch for computing contrastive loss
                    idx_i, idx_j = np.random.choice(img_inputs[0].shape[0], 2)
                    loss_contrastive = contrastive_loss(enc_out, idx_i, idx_j, margin=args.margin)
                    writer.add_scalar('train/loss_contrastive', loss_contrastive, iter_num)
                    loss_g += args.lambda_triplet*loss_contrastive

                # compute GAN loss
                if args.lambda_GAN > 0:
                    loss_gan_d = 0
                    loss_gan_g = 0
                    for contrast, img_output, img_target in zip(contrast_outputs, img_outputs, img_targets):
                        loss_gan_d_i = model_D.calc_dis_loss(img_output.detach(), img_target, contrast)
                        loss_gan_g_i = model_D.calc_gen_loss(img_output, contrast)
                        loss_gan_d += loss_gan_d_i
                        loss_gan_g += loss_gan_g_i
                        writer.add_scalar(f'train/loss_gan_d_{contrast}', loss_gan_d_i, iter_num)
                        writer.add_scalar(f'train/loss_gan_g_{contrast}', loss_gan_g_i, iter_num)

                    loss_gan_d = loss_gan_d/len(contrast_outputs)
                    loss_gan_g = loss_gan_g/len(contrast_outputs)

                    writer.add_scalar(f'train/loss_gan_d', loss_gan_d, iter_num)
                    writer.add_scalar(f'train/loss_gan_g', loss_gan_g, iter_num)
                    loss_meter_gan_d.update(loss_gan_d.item())
                    loss_meter_gan_g.update(loss_gan_g.item())
                    loss_g += args.lambda_GAN*loss_gan_g

                if args.lambda_perceptual > 0:
                    loss_vgg = recon_loss(
                        img_outputs, img_targets, criterion_perceptual
                    )
                    loss_g += args.lambda_perceptual * loss_vgg
                    loss_meter_perceptual.update(loss_vgg.item())

                if args.lambda_ssim > 0:
                    loss_ssim = recon_loss(img_outputs, img_targets, criterion_ssim)
                    loss_ssim = (1 - loss_ssim)
                    loss_g += args.lambda_ssim * loss_ssim
                    loss_meter_ssim.update(loss_ssim.item())

                # update generator
                opt_G.zero_grad()
                loss_g.backward()
                opt_G.step()
                lr_scheduler_g.step_update(epoch_num * len(trainloader) + i_batch)
                loss_meter_g.update(loss_g.item())
                loss_meter_rec.update(loss_output_rec.item())

                # update discriminator
                if args.lambda_GAN > 0:
                    opt_D.zero_grad()
                    loss_d = args.lambda_GAN*loss_gan_d
                    loss_d.backward()
                    opt_D.step()
                    lr_scheduler_d.step_update(epoch_num * len(trainloader) + i_batch)
                    writer.add_scalar('info/lr_d', get_lr(opt_D), iter_num)

                logging.info(f'iteration {iter_num} : loss_g: {loss_g.item()}')
                writer.add_scalar('info/lr_g', get_lr(opt_G), iter_num)
                writer.add_scalar('train/loss_g', loss_g, iter_num)

                if iter_num % args.vis_freq == 0:
                    writer.add_image('train/inputs', make_image_grid(img_inputs), iter_num)
                    writer.add_image('train/targets', make_image_grid(img_targets), iter_num)
                    writer.add_image('train/outputs', make_image_grid(img_outputs), iter_num)
                    if args.lambda_seg > 0:
                        writer.add_image('train/seg_out', make_seg_grid(torch.sigmoid(seg_out)), iter_num)
                        writer.add_image('train/seg_gt', make_seg_grid(seg_gt), iter_num)
                    if args.lambda_self > 0:
                        writer.add_image('train/inputs_recon', make_image_grid(img_inputs_recon), iter_num)

        # log epoch loss
        writer.add_scalar('epoch/loss_g', loss_meter_g.avg, epoch_num)
        writer.add_scalar('epoch/loss_output_rec', loss_meter_rec.avg, epoch_num)
        if args.lambda_GAN > 0:
            writer.add_scalar('epoch/loss_gan_g', loss_meter_gan_g.avg, epoch_num)
            writer.add_scalar('epoch/loss_gan_d', loss_meter_gan_d.avg, epoch_num)
        if args.lambda_seg > 0:
            writer.add_scalar('epoch/loss_seg', loss_meter_seg.avg, epoch_num)

        state_dict = {'G': model_G.module.state_dict(), 'epoch': epoch_num, 'opt_G': opt_G.state_dict(),
                      'lr_scheduler_g': lr_scheduler_g.state_dict()}
        if args.lambda_GAN > 0:
            state_dict['D'] = model_D.module.state_dict()
            state_dict['opt_d'] = opt_D.state_dict()
            state_dict['lr_scheduler_d'] = lr_scheduler_d.state_dict()

        if args.lambda_perceptual > 0:
            writer.add_scalar('epoch/loss_perceptual', loss_meter_perceptual.avg, epoch_num)

        # test on validation set
        if epoch_num % args.val_freq == 0:
            if args.zero_gad:
                val_combination = [[0, 2, 3]]
            elif args.k == 3:
                val_combination = input_combination_3
            else:
                val_combination = input_combination
            val_metrics, val_metrics_seg, _ = evaluator_brats(args, G, val_combination, split='val',
                                                                               seg=args.lambda_seg > 0, masked=args.lambda_seg > 0)
            performance_mae = 0
            performance_mse = 0
            performance_ssim = 0
            performance_psnr = 0
            for i in range(len(val_combination)):
                output_combination = list(set(range(4)) - set(val_combination[i]))
                if args.lambda_seg > 0:
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/seg_ET', val_metrics_seg[i]['ET'], epoch_num)
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/seg_TC', val_metrics_seg[i]['TC'], epoch_num)
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/seg_WT', val_metrics_seg[i]['WT'], epoch_num)
                for j in range(len(val_metrics[i])):
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/ssim_{output_combination[j]}', val_metrics[i][j]['ssim'], epoch_num)
                    performance_ssim += val_metrics[i][j]['ssim']
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/mae_{output_combination[j]}', val_metrics[i][j]['mae'], epoch_num)
                    performance_mae += val_metrics[i][j]['mae']
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/psnr_{output_combination[j]}', val_metrics[i][j]['psnr'], epoch_num)
                    performance_psnr += val_metrics[i][j]['psnr']
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/mse_{output_combination[j]}', val_metrics[i][j]['mse'], epoch_num)
                    performance_mse += val_metrics[i][j]['mse']
            if performance_mse < best_performance_mse:
                best_performance_mse = performance_mse
                save_path = os.path.join(snapshot_path, 'best_mse.pth')
                state_dict['mse'] = best_performance_mse
                torch.save(state_dict, save_path)
            if performance_mae < best_performance_mae:
                best_performance_mae = performance_mae
                save_path = os.path.join(snapshot_path, 'best_mae.pth')
                state_dict['mae'] = best_performance_mae
                torch.save(state_dict, save_path)
            if performance_ssim >= best_performance_ssim:
                best_performance_ssim = performance_ssim
                save_path = os.path.join(snapshot_path, 'best_ssim.pth')
                state_dict['ssim'] = best_performance_ssim
                torch.save(state_dict, save_path)
            if performance_psnr >= best_performance_psnr:
                best_performance_psnr = performance_psnr
                save_path = os.path.join(snapshot_path, 'best_psnr.pth')
                state_dict['psnr'] = best_performance_psnr
                torch.save(state_dict, save_path)

        # save ckpt every epoch
        state_dict['ssim'] = best_performance_ssim
        state_dict['mse'] = best_performance_mse
        state_dict['mae'] = best_performance_mae
        state_dict['psnr'] = best_performance_psnr
        # Save model on GCS bucket
        save_path = store_model(model_dir, epoch_num)
        torch.save(state_dict, save_path)
        logging.info("save model to {}".format(save_path))
        copy_model(gcs_model_path=save_path)

    writer.close()
    return "Training Finished!"
