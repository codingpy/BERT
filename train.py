import os

import re

import collections

import pandas as pd

import numpy as np

import tqdm

import tabulate

import transformers

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

from apex import amp

from apex.parallel import DistributedDataParallel


# pretraining uses the same configuration as google-research/bert's

def find_alphas(df):
    labels = df.iloc[:, 1:].columns

    alphas = []

    for label in labels:
        tgt = df[label]

        fqs = []

        for i in range(4):
            num = (tgt == i - 2).sum()

            if num != 0:
                fq = 1 / num
            else:
                fq = 0

            fqs.append(fq)

        alphas.append(fqs)
  
    for i, alpha in enumerate(alphas):
        total = sum(alpha)

        for j, a in enumerate(alpha):
            alpha[j] = a / total

    return torch.tensor(alphas)


def evaluate(net, loader):
    net.eval()

    # N x 7 x 3 x

    dat = []

    ids = []

    for tensors in loader:
        x, losses = net(*[x.cuda() for x in tensors])

        ids += list(tensors[0])

        # 7 x N x 4 D

        x = x.cpu()

        x = x.unbind(dim=-2)

        # 7 D

        losses = losses.cpu()

        # 7 x N D

        y = tensors[-1].T

        dat.append(
            # 7 x 3 x

            list(zip(x, y, losses))
        )

    # 7 x 3 x N x

    dat = [list(zip(*x)) for x in zip(*dat)]

    probs_all = []

    # 7 x N D

    preds_all = []

    label_all = []

    losses = []

    accs = []

    f1s = []

    for x, y, losses_per_domain in dat:
        # N x 4 D

        x = torch.cat(x)

        probs = F.softmax(x, dim=-1)

        # N D

        y = torch.cat(y)

        # 0 D

        loss = torch.stack(losses_per_domain).mean()

        # N D

        c = probs.argmax(dim=-1)

        acc = torch.mean((c == y).float())

        f1 = macroF1(F.one_hot(c), F.one_hot(y))

        # tensor -> scalar

        loss = loss.item()

        acc = acc.item()

        if not isinstance(f1, float):
            f1 = f1.item()

        probs_all.append(probs)

        preds_all.append(c)

        label_all.append(y)

        losses.append(loss)

        accs.append(acc)

        f1s.append(f1)

    # N x 7 D

    preds_all = torch.stack(preds_all, dim=-1)

    return ids, probs_all, preds_all, label_all, losses, accs, f1s


def main(data_path, path=None, epoch=5, val_splits=1500, val_steps=50):
    # prepare data

    df = pd.read_csv(data_path)

    alphas = find_alphas(df)

    x = df.iloc[:, 0].to_list()

    # -2 -> 0
    # -1 -> 1
    #  0 -> 2
    #  1 -> 3

    annotations = df.iloc[:, 1:]

    y = annotations.values + 2

    labels = annotations.columns

    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')

    x = tokenizer.batch_encode_plus(x, pad_to_max_length=True, return_tensors='pt')

    y = torch.tensor(y)

    dataset_whole = torch.utils.data.TensorDataset(x['input_ids'], x['attention_mask'], x['token_type_ids'], y)

    # training data (distributed)

    dataset = torch.utils.data.Subset(dataset_whole, range(val_splits, len(dataset_whole)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=24, sampler=sampler)

    # validation data

    dataset = torch.utils.data.Subset(dataset_whole, range(0, val_splits))

    loader_val = torch.utils.data.DataLoader(dataset, batch_size=10)

    # gpu

    torch.backends.cudnn.benchmark = True

    net = SentimentAnalysis(alphas, path)

    net.cuda()

    # per-layer learning rates

    groups = []

    for i in range(12):
        groups += [
            {'params': [], 'lr': 2e-5 * 0.95 ** (11-i), 'weight_decay': .00},
            {'params': [], 'lr': 2e-5 * 0.95 ** (11-i), 'weight_decay': .01},
        ]

    for n, p in net.named_parameters():
        i = 11

        ma = re.search('encoder.layer.(\d+)', n)

        if ma:
            i = int(ma[1])

        j = 1

        if 'bias' in n:
            j = 0

        groups[2 * i + j]['params'].append(p)

    optimizer = torch.optim.AdamW(groups)

    # slanted triangular learning rates

    T = epoch * len(loader)

    cut = 0.1 * T

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: t / cut if t < cut else (T - t) / (T - cut))

    # fp16

    net, optimizer = amp.initialize(net, optimizer)

    net = DistributedDataParallel(net)

    if rank == 0:
        i = 0

        trainlosses = []

        evallosses_hist = []

        evalloss_best = None

        writer = SummaryWriter()

    for j in range(epoch):
        if rank == 0:
            loader = tqdm.tqdm(loader, desc=f'Epoch {j}')

        for tensors in loader:
            # training step

            net.train()

            x, losses = net(*[x.cuda() for x in tensors])

            loss = losses.mean()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

            optimizer.step()

            optimizer.zero_grad()

            scheduler.step()

            if rank == 0:
                losses = losses.tolist()

                trainlosses.append(losses)

                if (i + 1) % val_steps == 0:
                    trainlosses = np.mean(trainlosses, axis=0)

                    # evaluation

                    with torch.no_grad():
                        ids, evalprobs_all, evalpreds_all, evallabel_all, evallosses, evalaccs, evalf1s = evaluate(net, loader_val)

                    texts = []

                    for seq_ids in ids:
                        text = tokenizer.decode(seq_ids, skip_special_tokens=True)

                        texts.append(text)

                    # save best

                    evallosses_hist.append(evallosses)

                    evallosses_hist = evallosses_hist[-5:]

                    moving_average = np.mean(evallosses_hist)

                    if evalloss_best is None or evalloss_best > moving_average:
                        evalloss_best = moving_average

                        # + 2 gpu [only accelerate training]
                        # + pretrained [maybe ineffective]
                        # + balanced loss

                        if i > cut:
                            torch.save(net.state_dict(), f'net-{i:04}-{evalloss_best:.8f}.pt')

                    # tensorboard

                    writer.add_scalar('lr', scheduler.get_lr()[0], i)

                    fields = ['unknown', 'negative', 'neutral', 'positive']

                    for label, evalprobs, evallabels, trainloss, evalloss, evalacc, evalf1 in zip(
                                labels, evalprobs_all, evallabel_all, trainlosses, evallosses, evalaccs, evalf1s):
                        tag_loss_dict = {
                            f'train': trainloss,
                            f'eval': evalloss,
                        }

                        writer.add_scalars(f'Loss/{label}', tag_loss_dict, i)

                        writer.add_scalar(f'Accuracy/{label}', evalacc, i)

                        writer.add_scalar(f'F1/{label}', evalf1, i)

                        for class_index in range(evalprobs.size(-1)):
                            writer.add_pr_curve(f'PR#{label}/{fields[class_index]}', evallabels == class_index, evalprobs[:, class_index], i)

                    data = postprocess(evalpreds_all - 2, labels)

                    # pretty

                    for key, val in data.items():
                        val_ = []

                        for v in val:
                            for s, t in [
                                    (-2, ' '),
                                    (-1, '-'),
                                    ( 0, '/'),
                                    ( 1, '+'),
                            ]:
                                if v == s:
                                    v = t

                                    break

                            val_.append(v)

                        data[key] = val_

                    data_ = collections.OrderedDict()

                    # markdown table

                    data_['review'] = texts

                    data_.update(data)

                    data = data_

                    tablestring = tabulate.tabulate(data, headers='keys', tablefmt='github')

                    if i > cut:
                        # reduces summary size

                        writer.add_text(f'EvalResults#step{i}', tablestring, i)

                    writer.add_scalars('Loss/total', {'train': np.mean(trainlosses), 'eval': np.mean(evallosses)}, i)

                    writer.add_scalar('Accuracy/total', np.mean(evalaccs), i)

                    writer.add_scalar('F1/total', np.mean(evalf1s), i)

                    trainlosses = []

                i += 1

    if rank == 0:
        torch.save(net.state_dict(), f'net.pt')


def macroF1(output, target):
    res = 0
    num = 0

    for x, y in zip(
            output.unbind(dim=-1),
            target.unbind(dim=-1),
    ):
        P = x > .5
        N = x < .5

        TP = P[y == 1].sum().float()
        FP = P[y == 0].sum().float()
        FN = N[y == 1].sum().float()

        if TP:
            P = TP / (TP+FP)
            R = TP / (TP+FN)
        else:
            P = 1

            if FP + FN == 0:
                R = 1
            else:
                R = 0

        res += 2*P*R / (P+R)

        num += 1

    res /= num
    return res


class SentimentAnalysis(nn.Module):
    def __init__(self, alphas, path=None):
        super().__init__()

        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-chinese')

        self.classifiers = nn.ModuleList()

        for i in range(7):
            self.classifiers.append(nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(768, 4),
            ))

        self.criterions = nn.ModuleList()

        for i, alpha in enumerate(alphas):
            # check tendency of `KPI` and loss, balance loss among tasks

            if i == 0:
                alpha /= 40 # overfit only for `experience`
            elif i == 1:
                alpha /= 2 # overfit for `speed`
            elif i == 2:
                alpha /= 16 # overfit for `guess`
            elif i == 5:
                alpha *= 2 # underfit for `service`
            elif i == 6:
                alpha *= 2 # underfit especially for `willing to consume again`

            loss = FocalLoss(alpha=alpha, gamma=2)

            self.criterions.append(loss)

        # load state dict

        if os.path.isfile(path):
            state_dict = torch.load(path, map_location='cpu')

            state_dict_ = collections.OrderedDict()

            for k, v in state_dict.items():
                state_dict_[k.replace('module.', '')] = v

            self.load_state_dict(state_dict_)

        # load transformers' model

        if os.path.isdir(path):
            self.bert = transformers.BertModel.from_pretrained(path)

    def forward(self, input_ids, attention_mask, token_type_ids, y=None):
        x = self.bert(input_ids, attention_mask, token_type_ids)

        # averaging

        h = x[0].mean(dim=1)

        # 7 x N x 4 D

        x = [clf(h) for clf in self.classifiers]

        losses = []

        if y is not None:
            # 7 x N D

            y = y.T

            for criterion, x_, y_ in zip(self.criterions, x, y):
                loss = criterion(x_, y_)

                losses.append(loss)

            # 7 D

            losses = torch.stack(losses)

        # N x 7 x 4 D

        x = torch.stack(x, dim=-2)

        return x, losses


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()

        self.xentropy = nn.CrossEntropyLoss(weight=alpha, reduction='none')

        self.gamma = gamma

    def forward(self, x, y):
        return torch.mean(
            (1 - F.softmax(x).gather(1, y.unsqueeze(1))) ** self.gamma * self.xentropy(x, y)
        )


def preprocess(tokenizer, x):
    x = tokenizer.batch_encode_plus(x, pad_to_max_length=True, return_tensors='pt')

    return x


def postprocess(x, labels):
    output = {}

    # 7 x N

    x = x.T

    for label, prediction in zip(labels, x):
        output[label] = prediction.tolist()

    return output


class Pipeline:
    def __init__(self, path, labels, alphas):
        self.net = SentimentAnalysis(alphas, path)

        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')

        self.device = 'cpu'

        if torch.cuda.is_available():
            self.device = 'cuda'

        self.net.to(self.device)

        self.net.eval()

        self.labels = labels

    def __call__(self, x):
        with torch.no_grad():
            # user-friendly

            wrapped = 0

            if not isinstance(x, list):
                x = [x,]

                wrapped = 1

            x = preprocess(self.tokenizer, x)

            input_ids = x['input_ids'].to(self.device)

            attention_mask = x['attention_mask'].to(self.device)

            token_type_ids = x['token_type_ids'].to(self.device)

            y, _ = self.net(input_ids, attention_mask, token_type_ids)

            y = y.argmax(dim=-1)

            x = postprocess(y - 2, self.labels)

            if wrapped:
                for k, v in x.items():
                    x[k] = v[0]

            return x


if __name__ == '__main__':
    # 2 gpu ->  6 min

    # fine-tune

    # cat train.py | ssh host 'cd workspace/; python3 -m torch.distributed.launch --nproc_per_node=2 -'

    dist.init_process_group('nccl')

    rank = dist.get_rank()

    with torch.cuda.device(rank):
        # underfit except for `experience`

        main('data.csv', path='experiments/1-/results', epoch=10)
