import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import time
import torch
from model import *
from model.layer import CustomEvaluator
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util import Logger, Bar, AverageMeter, accuracy, load_dataset, warp_decay, split_params, init_config
from model.model_setting import rate_model_setting, bptt_model_setting, init_model
from spikingjelly.activation_based import functional
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def train(train_ldr, optimizer, model, evaluator, args):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if dist.get_rank() == 0:
        bar = Bar('Processing', max=len(train_ldr))

    for idx, (ptns, labels) in enumerate(train_ldr):
        device = next(model.parameters()).device
        ptns, labels = ptns.to(device), labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        functional.reset_net(model.module)
        init_model(model.module)
        if model.module.step_mode == 's':
            out_spikes = []
            for t in range(args.T):
                out = model(ptns)
                out_spikes.append(out)
            output = torch.stack(out_spikes, dim=0)
            avg_fr = output.mean(dim=0)
        else:
            in_data, _ = torch.broadcast_tensors(ptns, torch.zeros((args.T,) + ptns.shape))
            in_data = in_data.reshape(-1, *in_data.shape[2:])
            avg_fr = model(in_data)

        loss = evaluator(avg_fr, labels)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(avg_fr.data, labels.data, topk=(1, 5))
        losses.update(loss.data.item(), ptns.size(0))
        top1.update(prec1.item(), ptns.size(0))
        top5.update(prec5.item(), ptns.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if dist.get_rank() == 0:
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=idx + 1,
                size=len(train_ldr),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    if dist.get_rank() == 0:
        bar.finish()

    return top1.avg, losses.avg


def test(val_ldr, model, evaluator, args):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    if dist.get_rank() == 0:
        bar = Bar('Processing', max=len(val_ldr))
    with torch.no_grad():
        for idx, (ptns, labels) in enumerate(val_ldr):
            device = next(model.parameters()).device
            ptns, labels = ptns.to(device), labels.to(device)

            data_time.update(time.time() - end)

            functional.reset_net(model.module)
            if model.module.step_mode == 's':
                out_spikes = []
                for t in range(args.T):
                    out = model(ptns)
                    out_spikes.append(out)
                output = torch.stack(out_spikes, dim=0)
                avg_fr = output.mean(dim=0)
            else:
                in_data, _ = torch.broadcast_tensors(ptns, torch.zeros((args.T,) + ptns.shape))
                in_data = in_data.reshape(-1, *in_data.shape[2:])
                avg_fr = model(in_data)

            loss = evaluator(avg_fr, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(avg_fr.data, labels.data, topk=(1, 5))
            losses.update(loss.data.item(), ptns.size(0))
            top1.update(prec1.item(), ptns.size(0))
            top5.update(prec5.item(), ptns.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if dist.get_rank() == 0:
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=idx + 1,
                    size=len(val_ldr),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
        if dist.get_rank() == 0:
            bar.finish()

        return top1.avg, losses.avg


def main():
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # set device, data type
    device, dtype = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float

    log = Logger(args, args.log_path)
    log.info_args(args)
    writer = SummaryWriter(args.log_path)

    train_data, val_data, num_class = load_dataset(args.dataset, args.data_path, cutout=args.cutout,
                                                   auto_aug=args.auto_aug)
    train_sampler = DistributedSampler(train_data, rank=rank)
    val_sampler = DistributedSampler(val_data, rank=rank)
    train_ldr = DataLoader(dataset=train_data, batch_size=args.train_batch_size // world_size, shuffle=False,
                           sampler=train_sampler,
                           pin_memory=True, num_workers=args.num_workers)
    val_ldr = DataLoader(dataset=val_data, batch_size=args.val_batch_size // world_size, shuffle=False,
                         sampler=val_sampler,
                         pin_memory=True, num_workers=args.num_workers)

    kwargs_spikes = {'v_reset': args.v_reset, 'thresh': args.thresh, 'decay': warp_decay(args.decay),
                     'detach_reset': args.detach_reset, "rate_flag": args.rate_flag}
    model = eval(args.arch + f'(num_classes={num_class}, **kwargs_spikes)')
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.rate_flag:
        rate_model_setting(model, time_step=args.T, step_mode=args.step_mode)
    else:
        bptt_model_setting(model, time_step=args.T, step_mode=args.step_mode)
    model.to(device, dtype)
    model = DistributedDataParallel(model, device_ids=[rank])

    params = split_params(model)
    params = [
        {'params': params[1], 'weight_decay': args.wd},
        {'params': params[2], 'weight_decay': 0}
    ]

    if args.optim.lower() == 'sgdm':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, amsgrad=False)
    else:
        raise NotImplementedError()

    evaluator = torch.nn.CrossEntropyLoss()
    evaluator = CustomEvaluator(evaluator, "rate" if args.rate_flag else "bptt", args.T)

    start_epoch = 0
    best_epoch = 0
    best_acc = 0.0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['best_net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['best_epoch']
        best_acc = state['best_acc']
        log.info('Load checkpoint from epoch {}'.format(start_epoch))
        log.info('Best accuracy so far {}.'.format(best_acc))
        log.info('Test the checkpoint: {}'.format(test(val_ldr, model, evaluator, args=args)))

    args.start_epoch = start_epoch
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.num_epoch)
    else:
        raise NotImplementedError()

    for epoch in range(start_epoch, args.num_epoch):
        train_acc, train_loss = train(train_ldr, optimizer, model, evaluator, args=args)
        if args.scheduler != 'None':
            scheduler.step()
        val_acc, val_loss = test(val_ldr, model, evaluator, args=args)

        if dist.is_available() and dist.is_initialized():
            tensor_train_acc = torch.tensor([train_acc], dtype=torch.float32).to(device)
            tensor_train_loss = torch.tensor([train_loss], dtype=torch.float32).to(device)
            tensor_val_acc = torch.tensor([val_acc], dtype=torch.float32).to(device)
            tensor_val_loss = torch.tensor([val_loss], dtype=torch.float32).to(device)

            dist.all_reduce(tensor_train_acc, op=dist.reduce_op.SUM)
            dist.all_reduce(tensor_train_loss, op=dist.reduce_op.SUM)
            dist.all_reduce(tensor_val_acc, op=dist.reduce_op.SUM)
            dist.all_reduce(tensor_val_loss, op=dist.reduce_op.SUM)

            aggregated_train_acc = tensor_train_acc.item()
            aggregated_train_loss = tensor_train_loss.item()
            aggregated_val_acc = tensor_val_acc.item()
            aggregated_val_loss = tensor_val_loss.item()

            world_size = dist.get_world_size()
            train_acc = aggregated_train_acc / world_size
            train_loss = aggregated_train_loss / world_size
            val_acc = aggregated_val_acc / world_size
            val_loss = aggregated_val_loss / world_size

        if dist.get_rank() == 0:
            log.info(
                'Epoch %03d: train loss %.5f, test loss %.5f, train acc %.5f, test acc %.5f, Saved custom_model..  with acc %.5f in the epoch %03d' % (
                    epoch, train_loss, val_loss, train_acc, val_acc, best_acc, best_epoch))
            # record in tensorboard
            writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss}, epoch + 1)
            writer.add_scalars('Acc', {'val': val_acc, 'train': train_acc}, epoch + 1)

            # save checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                state = {
                    'best_acc': best_acc,
                    'best_epoch': epoch,
                    'best_net': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(args.log_path, 'model_weights.pth'))


if __name__ == '__main__':
    from config.config import args

    dist.init_process_group(backend='nccl')
    init_config(args)
    main()
