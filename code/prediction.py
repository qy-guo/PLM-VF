import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import argparse

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'GPU' if device.type == 'cuda' else 'CPU'} Available")

    model = torch.jit.load(args.model_path)
    model.to(device)
    model.eval()

    X_test = torch.load(args.input_path)
    X_test = X_test.to(device)

    with torch.no_grad():
        outputs, *_ = model(X_test)
        preds = (outputs > args.threshold).int()

    df = pd.DataFrame(preds.cpu().numpy(), columns=['Predictions'])
    df.to_csv(args.output_path)
    print(f"Results saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run prediction.')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file.')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the inputs file.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the predictions CSV file.')
    parser.add_argument('-p', '--threshold', type=float, default=0.5, help='Threshold for predictions (default: 0.5).')
    args = parser.parse_args()
    main(args)

