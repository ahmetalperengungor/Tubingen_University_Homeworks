import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class HW9NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HW9NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        u1 = self.fc1(x)
        z1 = self.relu(u1)
        u2 = self.fc2(z1)
        y = u2 
        return y

# I preferred to use the sizes from Q2
model = HW9NN(input_size=4, hidden_size=4, output_size=3)
temp_example_input = torch.randn(1, 4)
writer = SummaryWriter('runs/hw9')
writer.add_graph(model, temp_example_input)
writer.close()
