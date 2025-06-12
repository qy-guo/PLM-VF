import pandas as pd
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'GPU' if device.type == 'cuda' else 'CPU'} Available")

    model = torch.jit.load(args.model_path)
    model.to(device)
    model.eval()

    test_label_data = torch.load(args.test_data_path)
    X_test = test_label_data[:, 1:]
    y_test = test_label_data[:, 0]

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=False)

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, *_ = model(inputs)
            all_probs.extend(outputs.tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend((outputs.data > args.threshold).int().tolist())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    spe = tn / (tn + fp)
    pre = precision_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Acc: {acc:.4f}, Recall: {recall:.4f}, Spe: {spe:.4f}, Pre: {pre:.4f}, Auroc: {auroc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

    df = pd.DataFrame({
        'Labels': all_labels,
        'Predictions': all_preds
    })
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run test.')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file.')
    parser.add_argument('-i', '--test_data_path', type=str, required=True, help='Path to the test data file.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the predictions CSV file.')
    parser.add_argument('-p', '--threshold', type=float, default=0.5, help='Threshold for predictions (default: 0.5).')
    args = parser.parse_args()
    main(args)

