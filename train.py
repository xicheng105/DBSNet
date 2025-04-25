import time
import os
import sys
import datetime
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from spikingjelly.activation_based import functional
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from utilities import LoadData, analyze_dataset, print_memory_usage
from models import build_model


# %% train_and_evaluate
def train_and_evaluate(model_t, train_loader_t, validation_loader_t, args_t, current_fold=0, total_folds=1):
    start_time_total = time.time()

    start_epoch = 0
    max_f1_score = -1
    previous_total_time = 0.0
    train_loss = 0.0
    train_acc = 0.0
    f1 = 0.0

    validation_loss_in_max_f1 = 0.0
    validation_acc_in_max_f1 = 0.0
    precision_in_max_f1 = 0.0
    recall_in_max_f1 = 0.0
    tp_rate_in_max_f1 = 0.0
    tn_rate_in_max_f1 = 0.0

    optimizer = torch.optim.Adam(model_t.parameters(), lr=args_t.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    main_output_dir = Path(args_t.output_path) / f"{args_t.dataset}_{args_t.model}"
    main_output_dir.mkdir(parents=True, exist_ok=True)

    subject_suffix = getattr(args_t, 'current_subject', 'TUSZ')
    sub_output_dir = main_output_dir / f"{main_output_dir.name}_sub-{subject_suffix[-2:]}"

    if args_t.model == 'EEG_DSNet':
        sub_output_dir = sub_output_dir.with_name(
            sub_output_dir.name + f'_{args_t.pooling_type}_{args_t.neuron_type}_step{args_t.T}'
        )
    sub_output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(sub_output_dir), purge_step=start_epoch)
    with open(sub_output_dir / 'args.txt', 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args_t))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    if args_t.resume and args_t.checkpoint is not None:
        checkpoint = torch.load(args_t.checkpoint, map_location='cpu')
        model_t.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_f1_score = checkpoint.get('max_f1_score', -1)
        previous_total_time = checkpoint.get('total_time_sec', 0.0)
        print(f"Resumed from {args_t.checkpoint}, starting from epoch {start_epoch}")

    for epoch in range(start_epoch, args_t.epochs):
        start_time = time.time()

        # Training
        model_t.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        for X, y in train_loader_t:
            optimizer.zero_grad()
            sample = X.to(args_t.device)
            label = y.to(args_t.device)

            out_fr = model_t(sample)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += torch.eq(out_fr.argmax(1), label).float().sum().item()
            if args_t.model_type == 'SNN':
                functional.reset_net(model_t)

        train_speed = train_samples / (time.time() - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        # validation
        model_t.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in validation_loader_t:
                sample = X.to(args_t.device)
                label = y.to(args_t.device)
                out_fr = model_t(sample)
                loss = criterion(out_fr, label)

                val_loss += loss.item() * label.numel()
                val_acc += torch.eq(out_fr.argmax(1), label).float().sum().item()

                all_preds.append(out_fr.argmax(1).cpu())
                all_labels.append(label.cpu())

                if args_t.model_type == 'SNN':
                    functional.reset_net(model_t)

        val_loss /= len(validation_loader_t.dataset)
        val_acc /= len(validation_loader_t.dataset)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        tp_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
        tn_rate = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Log extra metrics
        writer.add_scalar('validation_loss', val_loss, epoch)
        writer.add_scalar('validation_acc', val_acc, epoch)
        writer.add_scalar('validation_f1', f1, epoch)
        writer.add_scalar('validation_precision', precision, epoch)
        writer.add_scalar('validation_recall', recall, epoch)

        if f1 > max_f1_score:
            max_f1_score = f1
            validation_loss_in_max_f1 = val_loss
            validation_acc_in_max_f1 = val_acc
            precision_in_max_f1 = precision
            recall_in_max_f1 = recall
            tp_rate_in_max_f1 = tp_rate
            tn_rate_in_max_f1 = tn_rate

            checkpoint = {
                'model': model_t.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_f1_score': max_f1_score,
                'total_time_sec': (time.time() - start_time_total) + previous_total_time
            }
            torch.save(checkpoint, os.path.join(sub_output_dir, 'checkpoint_max.pth'))

        print(
            f'epoch = {epoch}\n'
            f'           train loss ={train_loss: .4f},      train acc ={train_acc: .4f}\n'
            f'      validation loss ={val_loss: .4f}, validation acc ={val_acc: .4f}, '
            f'    F1-Score = {f1:.4f}'
        )
        print(
            f'            Precision = {precision:.4f},         Recall = {recall:.4f}, max F1-score ={max_f1_score: .4f}'
        )
        print(
            f'          train speed = {int(train_speed)} samples/s'
        )
        temple_time = datetime.timedelta(seconds=(time.time() - start_time) * (args_t.epochs - epoch))
        print(
            f'estimated finish time = {(datetime.datetime.now() + temple_time).strftime("%Y-%m-%d %H:%M:%S")}'
        )
        print_memory_usage()
        print('')
        # sys.exit(0)

    end_time_total = time.time()
    single_fold_time_sec = (end_time_total - start_time_total) + previous_total_time
    remaining_folds = total_folds - (current_fold + 1)
    estimated_total_remaining_sec = single_fold_time_sec * remaining_folds

    estimated_total_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_total_remaining_sec)
    print(f"Estimated TOTAL finish time for all folds = {estimated_total_finish_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    csv_path = main_output_dir / 'training_results.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            'dataset', 'subject', 'model', 'neuron_type', 'T','train_loss', 'train_acc', 'validation_loss_in_max_f1',
            'validation_acc_in_max_f1','max_f1_score', 'f1_score', 'precision_in_max_f1', 'recall_in_max_f1',
            'tp_rate_in_max_f1', 'tn_rate_in_max_f1', 'total_time_sec', 'log_path'
        ])

    result = {
        'dataset': args_t.dataset,
        'subject': getattr(args_t, 'current_subject', 'TUSZ'),
        'model': args_t.model,
        'neuron_type': args_t.neuron_type if args_t.model_type == 'SNN' else 'None',
        'T': args_t.T if args_t.model_type == 'SNN' else 'None',
        'train_loss': round(train_loss, 4),
        'train_acc': round(train_acc, 4),
        'validation_loss_in_max_f1': round(validation_loss_in_max_f1, 4),
        'validation_acc_in_max_f1': round(validation_acc_in_max_f1, 4),
        'max_f1_score': round(max_f1_score, 4),
        'f1_score': round(f1, 4),
        'precision_in_max_f1': round(precision_in_max_f1, 4),
        'recall_in_max_f1': round(recall_in_max_f1, 4),
        'tp_rate_in_max_f1': round(tp_rate_in_max_f1, 4),
        'tn_rate_in_max_f1': round(tn_rate_in_max_f1, 4),
        'total_time_sec': int(single_fold_time_sec),
        'log_path': str(sub_output_dir)
    }

    df = df[~((df['dataset'] == result['dataset']) & (df['subject'] == result['subject']))]
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(csv_path, index=False)


# %% main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seizure EEG detection.')
    parser.add_argument(
        '--device',
        choices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5'],
        default='cuda:2',
        help='Which GPU to use.'
    )
    parser.add_argument(
        '--dataset',
        choices=['TUSZ', 'Siena', 'CHB_MIT'],
        default='Siena',
        help='Which dataset to use.'
    )
    parser.add_argument(
        '--model_type',
        default='ANN',
        choices=['ANN', 'SNN'],
        help='Which model type to use.'
    )
    parser.add_argument(
        '--model',
        default='EEGNet',
        choices=['EEGNet', 'EEGNeX', 'EEG_DSNet', 'CSNN', 'SCNet', 'LENet'],
        help='Which model to use.'
    )
    # ========================== Parameters just for EEG_DSNet ==========================
    parser.add_argument(
        '--neuron_type',
        default='AQIFNode',
        choices=['IFNode', 'LIFNode', 'IzhikevichNode', 'AQIFNode'],
        help='Neuron type to use.'
    )
    parser.add_argument(
        '--T',
        default=5,
        type=int,
        help='simulating time-steps.'
    )
    parser.add_argument(
        '--pooling_type',
        default='AM',
        choices=['AM', 'MA', 'AA', 'MM'],
        type=str,
        help='Pooling type.'
    )
    # ==================================================================================
    parser.add_argument(
        '--epochs',
        default=500,
        type=int,
        help='Number of epochs.'
    )
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='Batch size.'
    )
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        type=float,
        help='Learning rate.'
    )
    parser.add_argument(
        '--workers',
        default=0,
        type=int,
        help='Number of data loading workers.(Total number of cpu is 96.)'
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='Random seed.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Whether to resume from the checkpoint.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to the checkpoint file to resume from.'
    )
    parser.add_argument(
        '--output_path',
        default="/data4/louxicheng/python_projects/paper_3/",
        type=str
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initializing data loader.
    file_path = "/data4/louxicheng/EEG_data/seizure/"
    if args.dataset == 'TUSZ':
        file_path = Path(file_path) / 'TUSZ' / 'v2.0.3' / 'processed' / '01_tcp_ar_slice_length_4_seconds'
        dataset_configuration = {"chunk_size": 1000, "num_electrodes": 22, "sampling_rate": 250}
    elif args.dataset == 'Siena':
        file_path = Path(file_path) / 'Siena' / 'processed'
        dataset_configuration = {"chunk_size": 1024, "num_electrodes": 19, "sampling_rate": 256}
    else:
        file_path = Path(file_path) / 'CHB_MIT' / 'processed'
        dataset_configuration = {"chunk_size": 1024, "num_electrodes": 18, "sampling_rate": 256}

    if args.model_type == 'SNN':
        file_path = Path(file_path) / 'normalized'

    # Leave-One-Subject-Out
    if args.dataset in ['Siena', 'CHB_MIT']:
        subject_list = sorted([d.name for d in file_path.iterdir() if d.is_dir() and d.name.startswith('sub')])
        print(f"Found {len(subject_list)} subjects: {subject_list}")

        csv_base = Path(args.output_path) / f"{args.dataset}_{args.model}"
        CSV_path = csv_base / "training_results.csv"
        if CSV_path.exists():
            DF = pd.read_csv(CSV_path)
            finished_subjects = set(DF['subject'].tolist())
        else:
            finished_subjects = set()

        for leave_out_idx, test_subject in enumerate(subject_list):
            if test_subject in finished_subjects:
                print(f"Skipping {test_subject}, already trained.")
                continue
            args.current_subject = test_subject
            train_subjects = [subj for subj in subject_list if subj != test_subject]

            train_dataset = LoadData(file_path, down_sampling=True, subjects=train_subjects)
            test_dataset = LoadData(file_path, down_sampling=False, subjects=[test_subject])

            train_loader = data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.workers,
                pin_memory=True
            )
            test_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.workers,
                pin_memory=True
            )

            print(
                f"\n========== [{args.dataset}] Fold {leave_out_idx + 1}/{len(subject_list)}: "
                f"Test Subject = {test_subject} =========="
            )
            analyze_dataset(train_dataset, dataset_name="Train")
            analyze_dataset(test_dataset, dataset_name="Test")

            model = build_model(args, dataset_configuration)

            train_and_evaluate(
                model,
                train_loader,
                test_loader,
                args,
                current_fold=leave_out_idx,
                total_folds=len(subject_list)
            )

    # TUSZ training one time.
    else:
        train_dataset_path = file_path / 'train'
        val_dataset_path = file_path / 'validation'

        train_dataset = LoadData(train_dataset_path, down_sampling=True)
        validation_dataset = LoadData(val_dataset_path, down_sampling=True)

        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True
        )
        validation_loader = data.DataLoader(
            dataset=validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True
        )

        print(f"\n========== [TUSZ] Start Training ==========")
        analyze_dataset(train_dataset, dataset_name="Train")
        analyze_dataset(validation_dataset, dataset_name="Test")
        model = build_model(args, dataset_configuration)
        train_and_evaluate(model, train_loader, validation_loader, args)
