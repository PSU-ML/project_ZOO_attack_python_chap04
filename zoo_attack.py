import torch
import random
from CNN.resnet import ResNet18
from load_data import load_data
#import torch.backends.cudnn as cudnn



# load the mnist dataset (images are resized into 32 * 32)
training_set, test_set = load_data(data='mnist')

# define the model
model = ResNet18(dim=1)

# detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# load the learned model parameters
model.load_state_dict(torch.load('./model_weights/cpu_model.pth'))

model.to(device)
model.eval()

# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''

def zoo_attack(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''

    return image

# test the performance of attack
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])



total = 0
success = 0
num_image = 10 # number of images to be attacked

for i, (images, labels) in enumerate(testloader):
    target_label = get_target(labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    total += 1

    #adv_image = zoo_attack(network=model, image=images, target=target_label)
    adv_image = zoo_attack(network=model, image=images, t_0=labels)
    adv_image = adv_image.to(device)
    adv_output = model(adv_image)
    _, adv_pred = adv_output.max(1)
    if adv_pred.item() != labels.item():
        success += 1

    if total >= num_image:
        break

print('success rate : %.4f'%(success/total))



