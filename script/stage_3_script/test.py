from torch import nn

class CNN(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)

def print_num_params(model):
    print(f"{'Layer':<40} {'Params':>10}")
    print("-" * 52)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total += num_params
            print(f"{name:<40} {num_params:>10}")
    print("-" * 52)
    print(f"{'Total':<40} {total:>10}")

print_num_params(CNN())