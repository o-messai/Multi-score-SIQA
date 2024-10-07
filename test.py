import torch
import torch.nn as nn
from network import My_Net
from load_dataset import SIQADataset
import numpy as np
from scipy import stats
import argparse
import yaml
import os
import matplotlib.pyplot as plt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_indexNum(dataset, config, index, status):
    test_ratio = config['test_ratio']
    train_ratio = config['train_ratio']
    trainindex = index[:int(train_ratio * len(index))]
    testindex = index[int((1 - test_ratio) * len(index)):]
    train_index = [] 
    test_index = [] 

    ref_ids = []
    ref_ids_path = f"./data/{dataset}/ref_ids_S.txt"
    if not os.path.exists(ref_ids_path):
        raise FileNotFoundError(f"Reference IDs file not found at {ref_ids_path}")
    
    with open(ref_ids_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ref_ids.append(float(line))
    ref_ids = np.array(ref_ids)

    for i in range(len(ref_ids)):
        if ref_ids[i] in trainindex:
            train_index.append(i)
        elif ref_ids[i] in testindex:
            test_index.append(i)
        else:
            print(f"Error in splitting data for index {i}")

    if status == 'train':
        selected_index = train_index
    elif status == 'test':
        selected_index = test_index
    else:
        raise ValueError("Status must be either 'train' or 'test'")

    return len(selected_index), selected_index

def main():
    parser = argparse.ArgumentParser(description='Evaluate the SIQA model on the test dataset.')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (e.g., best_W1.pth)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--dataset", type=str, default="Waterloo_1", choices=["Waterloo_1", "Waterloo_2"], help="Dataset to evaluate on")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'#==> Using device: {device}')

    # Initialize and load the model
    model = My_Net().to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f'#==> Loaded model from {args.model_path}')

    # Prepare the test dataset and loader
    index = []
    if args.dataset == "Waterloo_1":
        index = list(range(1, 7))
    elif args.dataset == "Waterloo_2":
        index = list(range(1, 11))
    else:
        raise ValueError("Unsupported dataset")

    testnum, test_indices = get_indexNum(args.dataset, config, index, "test")
    test_dataset = SIQADataset(args.dataset, config, index, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # Evaluation metrics
    y_pred = []
    y_test = []
    criterion = nn.L1Loss()
    total_loss = 0

    with torch.no_grad():
        for i, (patchesL, patchesR, (label, label_L, label_R)) in enumerate(test_loader):
            patchesL = patchesL.to(device)
            patchesR = patchesR.to(device)
            label = label.to(device)

            outputs = model(patchesL, patchesR)[0]  # Assuming Q_index=0 for Global Quality
            loss = criterion(outputs, label)
            total_loss += loss.item()

            y_pred.extend(outputs.cpu().numpy())
            y_test.extend(label.cpu().numpy())

    # Convert lists to numpy arrays
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    average_loss = total_loss / (i + 1)

    # Calculate metrics
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    KROCC = stats.kendalltau(y_pred, y_test)[0]
    RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))

    # Output results
    result_str = (
        f"Final test Results: loss={average_loss:.3f} "
        f"SROCC={SROCC:.3f} PLCC={PLCC:.3f} "
        f"KROCC={KROCC:.3f} RMSE={RMSE:.3f}"
    )
    print(result_str)

    # Save results to a file
    ensure_dir('results')
    results_path = 'results/test_results.txt'
    with open(results_path, 'a+') as f:
        f.write(result_str + '\n')
    print(f'Results saved to {results_path}')

    # Optional: Save scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_test, c="blue", alpha=0.5)
    plt.xlabel("Predicted Quality Score")
    plt.ylabel("Ground Truth MOS")
    plt.title("MOS vs Predicted Quality Score")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
    plt.grid(True)
    plt.tight_layout()
    plot_path = 'results/mos_vs_pred_test.png'
    plt.savefig(plot_path)
    plt.close()
    print(f'Scatter plot saved to {plot_path}')

if __name__ == '__main__':
    main()
