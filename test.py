import torch
import argparse
import torch.utils.data as data

from spikingjelly.activation_based import functional
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from pathlib import Path
from utilities import LoadData, analyze_dataset
from models import build_model


# %% test
def evaluate(net, evaluate_loader, arguments):
    net.eval()
    test_samples = 0
    test_acc = 0.0
    test_prediction = []
    test_labels = []

    with torch.no_grad():
        for sample, label in evaluate_loader:
            sample = sample.to(arguments.device)
            label = label.to(arguments.device)

            output = net(sample)
            test_samples += label.numel()
            test_acc += torch.eq(output.argmax(1), label).float().sum().item()
            test_prediction.extend(output.argmax(1).cpu().numpy())
            test_labels.extend(label.cpu().numpy())

            if arguments.model_type == 'SNN':
                functional.reset_net(net)

    test_acc /= test_samples
    test_F1 = f1_score(test_labels, test_prediction, average='binary', zero_division=0)
    test_recall = recall_score(test_labels, test_prediction, average='binary', zero_division=0)
    cm = confusion_matrix(test_labels, test_prediction)
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    test_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    results = (
        f"Test accuracy: {test_acc * 100:.2f}%\n"
        f"Test F1-score: {test_F1:.4f}\n"
        f"Test recall: {test_recall:.4f}\n"
        f"Test specificity: {test_specificity:.4f}"
    )

    print(results)


# %% main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument(
        '--device',
        choices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5'],
        default='cuda:4',
        help='Which GPU to use.'
    )
    parser.add_argument(
        '--dataset',
        choices=['TUSZ', 'Siena', 'CHB_MIT'],
        default='TUSZ',
        help='Which dataset to use.'
    )
    parser.add_argument(
        '--model_type',
        default='SNN',
        choices=['ANN', 'SNN'],
        help='Which model type to use.'
    )
    parser.add_argument(
        '--model',
        default='EEG_DSNet',
        choices=['EEGNet', 'EEGNeX', 'EEG_DSNet', 'CSNN', 'SCNet', 'LENet'],
        help='Which model to use.'
    )
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='Batch size.'
    )
    parser.add_argument(
        '--model_path',
        default="/data4/louxicheng/python_projects/paper_3/",
        type=str
    )
    parser.add_argument(
        '--workers',
        default=0,
        type=int,
        help='Number of data loading workers.(Total number of cpu is 96.)'
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
        default=10,
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
    args = parser.parse_args()

    # Initializing data loader.
    dataset_configuration = {"chunk_size": 1000, "num_electrodes": 22, "sampling_rate": 250}
    file_path = Path("/data4/louxicheng/EEG_data/seizure/TUSZ/v2.0.3/processed/01_tcp_ar_slice_length_4_seconds/")
    if args.model_type == 'SNN':
        file_path = Path(file_path) / 'normalized'
    test_dataset_path = file_path / 'test'

    test_dataset = LoadData(test_dataset_path, down_sampling=False)
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True
    )
    analyze_dataset(test_dataset, dataset_name="Test")

    model = build_model(args, dataset_configuration)
    if args.model == "EEG_DSNet":
        checkpoint_path = (
                Path(args.model_path) /
                f"{args.dataset}_{args.model}" /
                f"{args.dataset}_{args.model}_sub-SZ_{args.pooling_type}_{args.neuron_type}_step{args.T}" /
                "checkpoint_max.pth"
        )
    else:
        checkpoint_path = (
                Path(args.model_path) /
                f"{args.dataset}_{args.model}" /
                "checkpoint_max.pth"
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])

    print(f"========== Test {checkpoint_path} ==========\n")
    evaluate(model, test_loader, args)
