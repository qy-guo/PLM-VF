import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
         # ProtT5 part
        self.pro_cnn1_3 = nn.Sequential(
            nn.Conv1d(1024, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pro_cnn1_5 = nn.Sequential(
            nn.Conv1d(1024, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pro_cnn1_7 = nn.Sequential(
            nn.Conv1d(1024, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # ESM-1b part
        self.esm1b_cnn1_3 = nn.Sequential(
            nn.Conv1d(1280, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.esm1b_cnn1_5 = nn.Sequential(
            nn.Conv1d(1280, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.esm1b_cnn1_7 = nn.Sequential(
            nn.Conv1d(1280, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # concat part
        self.maxpool = nn.MaxPool1d(kernel_size=9, stride=1, padding=4)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*6, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        pro = inputs[:, :, :1024]
        esm1b = inputs[:, :, 1024:1024+1280]
        # ProtT5
        pro = pro.permute(0, 2, 1)
        pro1 = self.pro_cnn1_3(pro)
        pro2 = self.pro_cnn1_5(pro)
        pro3 = self.pro_cnn1_7(pro)
        pro1 = self.maxpool(pro1.permute(0, 2, 1))
        pro2 = self.maxpool(pro2.permute(0, 2, 1))
        pro3 = self.maxpool(pro3.permute(0, 2, 1))
        pro_out = torch.concat([pro1, pro2, pro3], dim=-1)
        # ESM-1b
        esm1b = esm1b.permute(0, 2, 1)
        esm1b1 = self.esm1b_cnn1_3(esm1b)
        esm1b2 = self.esm1b_cnn1_5(esm1b)
        esm1b3 = self.esm1b_cnn1_7(esm1b)
        esm1b1 = self.maxpool(esm1b1.permute(0, 2, 1))
        esm1b2 = self.maxpool(esm1b2.permute(0, 2, 1))
        esm1b3 = self.maxpool(esm1b3.permute(0, 2, 1))
        esm1b_out = torch.concat([esm1b1, esm1b2, esm1b3], dim=-1)
        # concat
        outputs = torch.concat([pro_out, esm1b_out], dim=-1)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = outputs.squeeze()
        return outputs
