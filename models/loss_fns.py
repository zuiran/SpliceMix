import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

    def forward(self, input, target):
        # input: bs, nc; the output of model without sigmoid
        # target: bs, nc; multi-hot format

        return self.loss_fn(input, target)
