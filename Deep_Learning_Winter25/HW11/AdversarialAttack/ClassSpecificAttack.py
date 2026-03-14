# @author Maximus Mutschler and Nathan Inkawhich

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration Parameters
image_to_attack_path = "data/image_to_attack.npy"
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda = True
learning_rate = 0.1
max_opt_steps_per_class = 1000
reg_parameter = 0.01
grad_clip = 0.05
reg_mode = "L12"  # choose from "None", "L1", "L2", "L12", "GradClip"

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# laod image to attack
np_image_to_attack, label_of_image_to_attack = numpy.load(image_to_attack_path, allow_pickle=True)
image_to_attack = torch.unsqueeze(torch.Tensor(np_image_to_attack), dim=0).to(device)

# plot image to attack
plt.imshow(np_image_to_attack.squeeze(), cmap="gray")
plt.title(f"True Class: {label_of_image_to_attack}")
plt.xticks([], [])
plt.yticks([], [])
# plt.show()

target_classes = list(range(10))
target_classes.remove(label_of_image_to_attack)


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout Layers
model.eval()


def reg_loss(data, image_to_attack, reg_parameter, mode):
    """
    :param data: copy of the image to attack on, which the optimizer uses as params
    :param image_to_attack: the original unchanged image
    :param reg_parameter:
    :param mode: either "None" for no regularization (return 0)
                or "L2" for L2 regularization
                or "L1" for L1 regularization
                or "L12" for L1 and L2 regularization with the same parameter
                :return: the regularization loss which will be later added to the network loss
    """
    # TODO implement this method
    if mode == "None":
        return torch.tensor(0.0).to(device)
    
    diff = data - image_to_attack
    
    if mode == "L1":
        return reg_parameter * torch.norm(diff, p=1)
    elif mode == "L2":
        return reg_parameter * torch.norm(diff, p=2)
    elif mode == "L12":
        return reg_parameter * (torch.norm(diff, p=1) + torch.norm(diff, p=2))
    else:
        return torch.tensor(0.0).to(device)



# List that holds the images of all succesfull attacks
attacked_images = []

for target_class in target_classes:
    ###
    #    TODO
    #    Use a copy of the image_to_attack as parameters.
    #    Then use SGD without momentum to change those until the correct class in predicted.
    #    add images that resemble a successful attack to attacked_images
    ###
    # Use a copy of the image_to_attack as parameters
    perturbed_image = image_to_attack.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([perturbed_image], lr=learning_rate)
    
    for step in range(max_opt_steps_per_class):
        optimizer.zero_grad()
        output = model(perturbed_image)
        pred = output.max(1, keepdim=True)[1]
        
        if pred.item() == target_class:
            attacked_images.append((perturbed_image.detach().cpu().numpy().squeeze(), target_class))
            print(f"Target {target_class}: converged in {step} steps")
            break
            
        # Target misclassification loss: minimize NLL for target class
        loss = F.nll_loss(output, torch.tensor([target_class]).to(device))
        loss += reg_loss(perturbed_image, image_to_attack, reg_parameter, reg_mode)
        
        loss.backward()
        
        # Gradient clipping if selected
        if reg_mode == "GradClip":
            torch.nn.utils.clip_grad_norm_([perturbed_image], grad_clip)
            
        optimizer.step()
        
        # Clamp to [0, 1] range
        with torch.no_grad():
            perturbed_image.clamp_(0, 1)

###
#    TODO
#    Todo plot all images in attacked_images with their corresponding classed in addition to the original image.
#    This should be on Figure
###
# Plot all images in attacked_images with their corresponding classes in addition to the original image.
plt.figure(figsize=(15, 6))
# Original image
plt.subplot(2, 5, 1)
plt.imshow(np_image_to_attack.squeeze(), cmap="gray")
plt.title(f"Original Class: {label_of_image_to_attack}")
plt.axis("off")

# Attacked images
for i, (img, target) in enumerate(attacked_images):
    plt.subplot(2, 5, i + 2)
    plt.imshow(img, cmap="gray")
    plt.title(f"Target: {target}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(f"plots/attacked_images_{reg_mode}.png")