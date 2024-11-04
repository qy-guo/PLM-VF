import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix
from model import Net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'GPU' if device.type == 'cuda' else 'CPU'} Available")

all_labels = []
all_probs = []
all_preds = []

batch_size = 2

model_path = 'model/model.pt'
model = Net()
model = torch.load(model_path)
model.to(device)
model.eval()

test_label_data = torch.load('data/test_data.pt')
X_test = test_label_data[:, 1:]
y_test = test_label_data[:, 0]

test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        all_probs.extend(outputs.tolist())
        all_labels.extend(labels.tolist())
        all_preds.extend((outputs.data > 0.5).int().tolist())

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
acc  = accuracy_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
spe= tn / (tn + fp)
pre = precision_score(all_labels, all_preds)
auroc = roc_auc_score(all_labels, all_probs)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
print(f"Acc: {acc:.4f}, Recall: {recall:.4f}, Spe: {spe:.4f}, Pre: {pre:.4f}, Auroc: {auroc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")



