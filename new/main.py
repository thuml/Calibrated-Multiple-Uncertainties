import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.append('.')
from model import DomainDiscriminator, Ensemble
from model import DomainAdversarialLoss, ImageClassifier, resnet50
import datasets
from datasets import esem_dataloader
from lib import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator, AccuracyCounter
from lib import ResizeImage
from lib import StepwiseLR, get_entropy, get_confidence, get_consistency, norm, single_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_tranform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    a, b, c = args.n_share, args.n_source_private, args.n_total
    common_classes = [i for i in range(a)]
    source_private_classes = [i + a for i in range(b)]
    target_private_classes = [i + a + b for i in range(c - a - b)]
    source_classes = common_classes + source_private_classes
    target_classes = common_classes + target_private_classes

    dataset = datasets.Office31
    train_source_dataset = dataset(root=args.root, data_list_file=args.source, filter_class=source_classes,
                                   transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, data_list_file=args.target, filter_class=target_classes,
                                   transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, data_list_file=args.target, filter_class=target_classes,
                          transform=val_tranform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    esem_iter1, esem_iter2, esem_iter3, esem_iter4, esem_iter5 = esem_dataloader(args, source_classes)

    # create model
    backbone = resnet50(pretrained=True)
    classifier = ImageClassifier(backbone, train_source_dataset.num_classes).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    esem = Ensemble(classifier.features_dim, train_source_dataset.num_classes).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    optimizer_esem = SGD(esem.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                         nesterov=True)
    lr_scheduler1 = StepwiseLR(optimizer_esem, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler2 = StepwiseLR(optimizer_esem, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler3 = StepwiseLR(optimizer_esem, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler4 = StepwiseLR(optimizer_esem, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler5 = StepwiseLR(optimizer_esem, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    optimizer_pre = SGD(esem.get_parameters() + classifier.get_parameters(), args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True)

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri, reduction='none').to(device)

    pretrain(esem_iter1, esem_iter2, esem_iter3, esem_iter4, esem_iter5, classifier,
             esem, optimizer_pre, args)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch

        train_esem(esem_iter1, classifier, esem, optimizer_esem, lr_scheduler1, epoch, args, index=1)
        train_esem(esem_iter2, classifier, esem, optimizer_esem, lr_scheduler2, epoch, args, index=2)
        train_esem(esem_iter3, classifier, esem, optimizer_esem, lr_scheduler3, epoch, args, index=3)
        train_esem(esem_iter4, classifier, esem, optimizer_esem, lr_scheduler4, epoch, args, index=4)
        train_esem(esem_iter5, classifier, esem, optimizer_esem, lr_scheduler5, epoch, args, index=5)

        source_class_weight = evaluate_source_common(val_loader, classifier, esem, source_classes, args)

        train(train_source_iter, train_target_iter, classifier, domain_adv, esem, optimizer,
              lr_scheduler, epoch, source_class_weight, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, esem, source_classes, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.3f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier, esem, source_classes, args)
    print("test_acc1 = {:3.3f}".format(acc1))


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, esem, optimizer: SGD,
          lr_scheduler: StepwiseLR, epoch: int, source_class_weight, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()
    esem.eval()

    end = time.time()
    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        # x = torch.cat((x_s, x_t), dim=0)
        # y, f = model(x)
        # y_s, y_t = y.chunk(2, dim=0)
        # f_s, f_t = f.chunk(2, dim=0)
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        with torch.no_grad():
            yt_1, yt_2, yt_3, yt_4, yt_5 = esem(f_t)
            confidece = get_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)
            consistency = get_consistency(yt_1, yt_2, yt_3, yt_4, yt_5)
            w_t = (1 - entropy + 1 - consistency + confidece) / 3
            w_s = torch.tensor([source_class_weight[i] for i in labels_s]).to(device)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t, w_s.detach(), w_t.to(device).detach())
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_esem(train_source_iter, model, esem, optimizer, lr_scheduler, epoch, args, index):
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs],
        prefix="Esem: [{}-{}]".format(epoch, index))

    model.eval()
    esem.train()

    for i in range(args.iters_per_epoch // 2):
        lr_scheduler.step()

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        with torch.no_grad():
            y_s, f_s = model(x_s)
        y_s = esem(f_s.detach(), index)

        loss = F.cross_entropy(y_s, labels_s)
        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (args.print_freq * 5) == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, esem, source_classes: list,
             args: argparse.Namespace) -> float:
    # switch to evaluate mode
    model.eval()
    esem.eval()

    all_confidece = list()
    all_consistency = list()
    all_entropy = list()
    all_indices = list()
    all_labels = list()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            output, f = model(images)
            values, indices = torch.max(F.softmax(output, -1), 1)

            yt_1, yt_2, yt_3, yt_4, yt_5 = esem(f)
            confidece = get_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)
            consistency = get_consistency(yt_1, yt_2, yt_3, yt_4, yt_5)

            all_confidece.extend(confidece)
            all_consistency.extend(consistency)
            all_entropy.extend(entropy)
            all_indices.extend(indices)
            all_labels.extend(labels)

    all_confidece = norm(torch.tensor(all_confidece))
    all_consistency = norm(torch.tensor(all_consistency))
    all_entropy = norm(torch.tensor(all_entropy))
    all_score = (all_confidece + 1 - all_consistency + 1 - all_entropy) / 3

    counters = AccuracyCounter(len(source_classes) + 1)
    for (each_indice, each_label, score) in zip(all_indices, all_labels, all_score):
        if each_label in source_classes:
            counters.add_total(each_label)
            if score >= args.threshold and each_indice == each_label:
                counters.add_correct(each_label)
        else:
            counters.add_total(-1)
            if score < args.threshold:
                counters.add_correct(-1)

    print('---counters---')
    print(counters.each_accuracy())
    print(counters.mean_accuracy())
    print(counters.h_score())

    return counters.mean_accuracy()


def evaluate_source_common(val_loader: DataLoader, model: ImageClassifier, esem, source_classes: list,
                           args: argparse.Namespace):
    temperature = 1
    # switch to evaluate mode
    model.eval()
    esem.eval()

    common = []
    target_private = []

    all_confidece = list()
    all_consistency = list()
    all_entropy = list()
    all_labels = list()
    all_output = list()

    source_weight = torch.zeros(len(source_classes)).to(device)
    cnt = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            # labels = labels.to(device)

            output, f = model(images)
            output = F.softmax(output, -1) / temperature
            yt_1, yt_2, yt_3, yt_4, yt_5 = esem(f)
            confidece = get_confidence(yt_1, yt_2, yt_3, yt_4, yt_5)
            entropy = get_entropy(yt_1, yt_2, yt_3, yt_4, yt_5)
            consistency = get_consistency(yt_1, yt_2, yt_3, yt_4, yt_5)

            all_confidece.extend(confidece)
            all_consistency.extend(consistency)
            all_entropy.extend(entropy)
            all_labels.extend(labels)

            for each_output in output:
                all_output.append(each_output)

            # for (each_output, each_score, label) in zip(output, score, labels):
            #     if each_score >= args.threshold:
            #         source_weight += each_output
            #         cnt += 1
            # if label in source_classes:
            #     common.append(each_score)
            # else:
            #     target_private.append(each_score)

    all_confidece = norm(torch.tensor(all_confidece))
    all_consistency = norm(torch.tensor(all_consistency))
    all_entropy = norm(torch.tensor(all_entropy))
    all_score = (all_confidece + 1 - all_consistency + 1 - all_entropy) / 3

    # args.threshold = torch.median(all_score)
    # print('threshold = {}'.format(args.threshold))

    for i in range(len(all_score)):
        if all_score[i] >= args.threshold:
            source_weight += all_output[i]
            cnt += 1
        if all_labels[i] in source_classes:
            common.append(all_score[i])
        else:
            target_private.append(all_score[i])

    hist, bin_edges = np.histogram(common, bins=10, range=(0, 1))
    print(hist)
    # print(bin_edges)

    hist, bin_edges = np.histogram(target_private, bins=10, range=(0, 1))
    print(hist)
    # print(bin_edges)

    source_weight = norm(source_weight / cnt)
    print('---source_weight---')
    print(source_weight)
    return source_weight


def pretrain(esem_iter1, esem_iter2, esem_iter3, esem_iter4, esem_iter5, model, esem, optimizer, args):
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs],
        prefix="Esem: [{}-{}]".format(0, 0))

    model.train()
    esem.train()

    for i in range(args.iters_per_epoch):
        x_s1, labels_s1 = next(esem_iter1)
        x_s1 = x_s1.to(device)
        labels_s1 = labels_s1.to(device)
        y_s1, f_s1 = model(x_s1)
        y_s1 = esem(f_s1, index=1)
        loss1 = F.cross_entropy(y_s1, labels_s1)

        x_s2, labels_s2 = next(esem_iter2)
        x_s2 = x_s2.to(device)
        labels_s2 = labels_s2.to(device)
        y_s2, f_s2 = model(x_s2)
        y_s2 = esem(f_s2, index=2)
        loss2 = F.cross_entropy(y_s2, labels_s2)

        x_s3, labels_s3 = next(esem_iter3)
        x_s3 = x_s3.to(device)
        labels_s3 = labels_s3.to(device)
        y_s3, f_s3 = model(x_s3)
        y_s3 = esem(f_s3, index=3)
        loss3 = F.cross_entropy(y_s3, labels_s3)

        x_s4, labels_s4 = next(esem_iter4)
        x_s4 = x_s4.to(device)
        labels_s4 = labels_s4.to(device)
        y_s4, f_s4 = model(x_s4)
        y_s4 = esem(f_s4, index=1)
        loss4 = F.cross_entropy(y_s4, labels_s4)

        x_s5, labels_s5 = next(esem_iter5)
        x_s5 = x_s5.to(device)
        labels_s5 = labels_s5.to(device)
        y_s5, f_s5 = model(x_s5)
        y_s5 = esem(f_s5, index=1)
        loss5 = F.cross_entropy(y_s5, labels_s5)

        cls_acc = accuracy(y_s1, labels_s1)[0]
        cls_accs.update(cls_acc.item(), x_s1.size(0))

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        losses.update(loss.item(), x_s1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (args.print_freq * 5) == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--n_share', type=int, default=10, help=" ")
    parser.add_argument('--n_source_private', type=int, default=10, help=" ")
    parser.add_argument('--n_total', type=int, default=31, help=" ")
    parser.add_argument('--threshold', type=float, default=0.5, help=" ")

    args = parser.parse_args()
    print(args)
    main(args)
