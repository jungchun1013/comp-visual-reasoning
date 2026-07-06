"""Train gCOG models across different architectures and generalization splits.

Usage:
    python scripts/gcog/train_gcog.py \
        --model transformer \
        --split distractor \
        --n-epochs 50 \
        --seed 42

Models: transformer, crossattn, gca, perceiver, multimodal, vit_crossattn, vit_gca
Splits: distractor, opsys, comptree, productive
"""

import argparse
import glob
import os
import sys
import time
import json

for p in glob.glob('/home/jungchun/miniconda3/lib/python3.*/site-packages'):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '/tmp/gcog')
import gcog.task.config as config
import gcog.task.dataset_generator as datagen
from gcog.models.transformer import (
    Transformer,
    SimpleCrossAttn,
    GatedCrossAttn,
    Perceiver,
    MultimodalTransformer,
    ViTCrossAttn,
    ViTGCA,
)

VIT_MODELS = {'vit_crossattn', 'vit_gca'}


class CachedViTDataset(Dataset):
    """Pre-caches ViT features for all samples at init time.

    Returns (rule_array, vit_tokens, target_idx, idx) where
    vit_tokens is (n_patches, embed_dim) from frozen DINOv2.
    """
    def __init__(self, base_dataset, vit_model, device, img_size=224, batch_size=64):
        import cv2
        self.cv2 = cv2

        self.rules = []
        self.targets = []
        self.img_size = img_size

        print(f"  Caching ViT features for {len(base_dataset)} samples...")
        all_imgs = []
        for i in range(len(base_dataset)):
            rule_array, stim_array, target_idx, _ = base_dataset[i]
            self.rules.append(rule_array)
            self.targets.append(target_idx)

            metatask = base_dataset.metatasks[i]
            if hasattr(base_dataset, 'distractor_range') and base_dataset.distractor_range:
                import random
                n_dist = random.choice(range(base_dataset.n_distractors + 1))
            else:
                n_dist = base_dataset.n_distractors
            objset = metatask.generate_objset(n_distractor=n_dist)
            img = objset.create_img(width=64, height=64)
            if img.ndim == 4:
                img = img[:, :, :, 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            all_imgs.append(img)

        all_features = []
        vit_model.eval()
        with torch.no_grad():
            for start in range(0, len(all_imgs), batch_size):
                batch = torch.tensor(np.array(all_imgs[start:start + batch_size]),
                                     dtype=torch.float32).to(device)
                out = vit_model.forward_features(batch)
                all_features.append(out['x_norm_patchtokens'].cpu())

        self.vit_features = torch.cat(all_features, dim=0)
        del all_imgs
        print(f"  Cached: {self.vit_features.shape}")

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx], self.vit_features[idx], self.targets[idx], idx


def collate_fn_cached(batch):
    rule_seq = []
    feat_seq = []
    target_seq = []
    idx_seq = []
    for rule, feat, target, idx in batch:
        rule_seq.append(torch.Tensor(rule))
        feat_seq.append(feat)
        target_seq.append(target)
        idx_seq.append(idx)
    rules = torch.nn.utils.rnn.pad_sequence(rule_seq, batch_first=True)
    feats = torch.stack(feat_seq)
    targets = torch.tensor(target_seq, dtype=torch.long)
    return rules, feats, targets, np.asarray(idx_seq)


def accuracy_score(outputs, targets):
    max_resp = torch.argmax(outputs, 1)
    acc = (max_resp == targets).float()
    return acc


def build_model(model_name, device, embedding_dim=256, nhead=1, dropout=0.1,
                lr=1e-4, max_tree_depth=100):
    kwargs = dict(
        embedding_dim=embedding_dim,
        num_hidden=embedding_dim,
        nhead=nhead,
        max_tree_depth=max_tree_depth,
        dropout=dropout,
        learning_rate=lr,
        lossfunc='NLL',
    )
    if model_name == 'transformer':
        model = Transformer(**kwargs)
    elif model_name == 'crossattn':
        model = SimpleCrossAttn(**kwargs)
    elif model_name == 'gca':
        model = GatedCrossAttn(**kwargs)
    elif model_name == 'perceiver':
        model = Perceiver(**kwargs)
    elif model_name == 'multimodal':
        model = MultimodalTransformer(**kwargs, nblocks=1)
    elif model_name == 'vit_crossattn':
        model = ViTCrossAttn(**kwargs, freeze_vit=True)
    elif model_name == 'vit_gca':
        model = ViTGCA(**kwargs, freeze_vit=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def build_datasets(split, ntrials=5000, nfeatures=10):
    common = dict(location=True, nfeatures=nfeatures)

    if split == 'distractor':
        train_ds = datagen.CompTreeDataset(
            tree_depth=1, n_distractors=5, distractor_range=True,
            ntrials=ntrials, **common)
        test_configs = {
            'test_5d': dict(tree_depth=1, n_distractors=5, distractor_range=False,
                            ntrials=ntrials, **common),
            'test_10d': dict(tree_depth=1, n_distractors=10, distractor_range=False,
                             ntrials=ntrials, **common),
            'test_20d': dict(tree_depth=1, n_distractors=20, distractor_range=False,
                             ntrials=ntrials, **common),
            'test_40d': dict(tree_depth=1, n_distractors=40, distractor_range=False,
                             ntrials=ntrials, **common),
        }
        test_datasets = {k: datagen.CompTreeDataset(**v) for k, v in test_configs.items()}
        max_depth = 1

    elif split == 'opsys':
        train_ds = datagen.OperatorSystematicity(
            n_distractors=5, distractor_range=True, subset=1, nfeatures=nfeatures, location=True)
        test_datasets = {
            'test_opsys': datagen.OperatorSystematicity(
                n_distractors=5, distractor_range=True, subset=2, nfeatures=nfeatures, location=True),
        }
        max_depth = 1

    elif split == 'comptree':
        train_ds = datagen.CompTreeDatasetSubset(
            tree_depth=3, n_distractors=5, distractor_range=True,
            ntrials=ntrials, subset=1, **common)
        test_datasets = {
            'test_comptree': datagen.CompTreeDatasetSubset(
                tree_depth=3, n_distractors=5, distractor_range=True,
                ntrials=ntrials, subset=2, **common),
        }
        max_depth = 3

    elif split == 'productive':
        train_ds = datagen.CompTreeDataset(
            tree_depth=3, n_distractors=5, distractor_range=True,
            ntrials=ntrials, **common)
        test_datasets = {
            'test_depth5': datagen.CompTreeDataset(
                tree_depth=5, n_distractors=5, distractor_range=True,
                ntrials=ntrials, **common),
            'test_depth7': datagen.CompTreeDataset(
                tree_depth=7, n_distractors=5, distractor_range=True,
                ntrials=ntrials, **common),
        }
        max_depth = 7

    else:
        raise ValueError(f"Unknown split: {split}")

    return train_ds, test_datasets, max_depth


def train_epoch(model, dataloader, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for rules, stims, targets, _ in dataloader:
        if isinstance(rules, list):
            continue

        rules = rules.float().to(device)
        stims = stims.float().to(device)
        targets = targets.to(device)

        model.zero_grad()
        outputs, _ = model(rules, stims, noise=False, dropout=False)
        loss = model.lossfunc(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.optimizer.step()

        total_loss += loss.item() * len(targets)
        acc = accuracy_score(outputs, targets)
        total_correct += acc.sum().item()
        total_samples += len(targets)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['transformer', 'crossattn', 'gca', 'perceiver', 'multimodal',
                                 'vit_crossattn', 'vit_gca'])
    parser.add_argument('--split', required=True,
                        choices=['distractor', 'opsys', 'comptree', 'productive'])
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ntrials', type=int, default=5000)
    parser.add_argument('--nfeatures', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            '/nfs/turbo/coe-chaijy/jungchun/vault/a-concept/comp-visual-reasoning',
            'outputs', 'gcog', f'{args.model}_{args.split}_seed{args.seed}')
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Building datasets: split={args.split}, ntrials={args.ntrials}, nfeatures={args.nfeatures}")
    train_ds, test_datasets, max_depth = build_datasets(
        args.split, ntrials=args.ntrials, nfeatures=args.nfeatures)
    print(f"Train: {len(train_ds)} samples | Test sets: {list(test_datasets.keys())}")

    use_vit = args.model in VIT_MODELS

    max_tree_depth = max(100, max_depth * 20)
    model = build_model(args.model, device,
                        embedding_dim=args.embedding_dim,
                        nhead=args.nhead,
                        dropout=args.dropout,
                        lr=args.lr,
                        max_tree_depth=max_tree_depth)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Params: {n_params:,} (trainable: {n_trainable:,})")

    if use_vit:
        print("Pre-caching ViT features...")
        train_ds = CachedViTDataset(train_ds, model.vit, device)
        test_datasets = {k: CachedViTDataset(v, model.vit, device)
                         for k, v in test_datasets.items()}
        collate_fn = collate_fn_cached
    else:
        collate_fn = datagen.collate_fn_rulepad

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=collate_fn)

    log = {
        'config': vars(args),
        'n_params': n_params,
        'epochs': [],
    }

    best_train_acc = 0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, device)
        dt = time.time() - t0

        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'time': dt,
            'test': {},
        }

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.n_epochs - 1:
            for test_name, test_ds in test_datasets.items():
                test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0,
                                         collate_fn=collate_fn)
                model.eval()
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for rules, stims, targets, _ in test_loader:
                        rules = rules.float().to(device)
                        stims = stims.float().to(device)
                        targets = targets.to(device)
                        outputs, _ = model(rules, stims)
                        acc = accuracy_score(outputs, targets)
                        test_correct += acc.sum().item()
                        test_total += len(targets)
                test_acc = test_correct / max(test_total, 1) * 100
                epoch_log['test'][test_name] = test_acc

            test_str = ' | '.join(f'{k}: {v:.1f}%' for k, v in epoch_log['test'].items())
            print(f"[Epoch {epoch:3d}/{args.n_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
                  f"{test_str} | {dt:.1f}s")
        else:
            print(f"[Epoch {epoch:3d}/{args.n_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | {dt:.1f}s")

        log['epochs'].append(epoch_log)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': model.optimizer.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))

    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': model.optimizer.state_dict(),
        'epoch': args.n_epochs - 1,
        'train_acc': train_acc,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Results saved to {args.output_dir}")
    print(f"Best train acc: {best_train_acc:.1f}%")
    if log['epochs'][-1]['test']:
        for k, v in log['epochs'][-1]['test'].items():
            print(f"Final {k}: {v:.1f}%")


if __name__ == '__main__':
    main()
