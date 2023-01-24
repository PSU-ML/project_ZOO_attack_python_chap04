import random

import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import grad

from CNN.resnet import ResNet18
from load_data import load_data

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
attack_hparams = {
    'lr': 0.01,
    'num_epochs': 1000,
    'initial_const': 0.001,
    'max_iteration': 1000,
    'transfer_param': 0.0,
}


def get_partial_loss(output, labels, labels_infhot=None):
    if labels_infhot is None:
        labels_infhot = torch.zeros_like(output).scatter_(1, labels.unsqueeze(1), float('inf'))

    # get the confidence score of the target class(real class)
    class_logits = output.gather(1, labels.unsqueeze(1)).squeeze(1)
    # get the maximum confidence score of all other classes
    other_logits = (output - labels_infhot).amax(dim=1)

    class_logits = torch.log(class_logits)
    other_logits = torch.log(other_logits)

    return class_logits - other_logits


def zoo_attack(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    # transfer_param: higher value means more tamper.
    transfer_param = attack_hparams['transfer_param']
    batch_size = len(image)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (image.ndim - 1))
    t_image = (image * 2).sub_(1).mul_(1 - 1e-6).atanh_()

    # setup regularization parameter c
    # lower_bound, upper_bound, use binary search to find the optimal c
    c = torch.full((1,), attack_hparams['initial_const'], device=device)
    lower_bound = torch.zeros_like(c)
    upper_bound = torch.full_like(c, 1e10)

    o_best_l2 = torch.full_like(c, float('inf'))
    o_best_adv = image.clone()
    o_adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)
    labels_onehot = None
    labels_infhot = None

    attack_epochs = attack_hparams['num_epochs']
    attack_iterations = attack_hparams['max_iterations']
    for epoch in range(attack_epochs):
        # setup the modifier and the optimizer
        modifier = torch.zeros_like(image, requires_grad=True)
        optimizer = optim.Adam([modifier], lr=attack_hparams['lr'])
        best_l2 = torch.full_like(c, float('inf'))
        adv_found = torch.zeros(1, device=device, dtype=torch.bool)

        # The last iteration (if we run many steps) repeat the search once.
        if (attack_epochs >= 10) and epoch == (attack_epochs - 1):
            c = upper_bound
        # record previous result to avoid stuck
        prev = float('inf')

        for i in range(attack_iterations):
            # generate the adversarial example
            adv_inputs = (torch.tanh(t_image + modifier) + 1) / 2
            # calculate similarity
            l2_squared = (adv_inputs - image).flatten(1).square().sum(1)
            l2 = l2_squared.detach().sqrt()
            # get model output
            output = model(adv_inputs)

            # in first run, setup the target variable, we need it to be in one-hot form for the loss function
            if epoch == 0 and i == 0:
                labels_onehot = torch.zeros_like(output).scatter_(1, labels.unsqueeze(1), 1)
                labels_infhot = torch.zeros_like(output).scatter_(1, labels.unsqueeze(1), float('inf'))

            # get prediction of the model
            prediction = (output + labels_onehot * transfer_param).argmax(1)
            # check if current image is an adversarial example
            is_adv = prediction != labels
            # get the most similar AE
            is_smaller = l2 < best_l2
            o_is_smaller = l2 < o_best_l2
            # in the case the image is an AE and the difference is low
            is_both = is_adv & is_smaller
            o_is_both = is_adv & o_is_smaller

            # update the current best similarity
            best_l2 = torch.where(is_both, l2, best_l2)
            adv_found.logical_or_(is_both)

            # update the global best and save the best AE
            o_best_l2 = torch.where(o_is_both, l2, o_best_l2)
            o_adv_found.logical_or_(is_both)
            o_best_adv = torch.where(batch_view(o_is_both), adv_inputs.detach(), o_best_adv)

            # calculate the loss
            partial_loss = get_partial_loss(output, labels, labels_infhot=labels_infhot)
            loss = l2_squared + c * (partial_loss + transfer_param).clamp_(min=0)

            # early stop
            abort_early = True
            if abort_early and i % (attack_iterations // 10) == 0:
                if (loss > prev * 0.9999).all():
                    break
                prev = loss.detach()

            optimizer.zero_grad(set_to_none=True)
            modifier.grad = grad(loss.sum(), modifier, only_inputs=True)[0]
            optimizer.step()

        # update c
        upper_bound[adv_found] = torch.min(upper_bound[adv_found], c[adv_found])
        adv_not_found = ~adv_found
        lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], c[adv_not_found])
        is_smaller = upper_bound < 1e9
        c[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
        c[(~is_smaller) & adv_not_found] *= 10

    # return the best adv
    return o_best_adv


# test the performance of attack
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)


def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss


def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])


total = 0
success = 0
num_image = 10  # number of images to be attacked

for i, (images, labels) in enumerate(testloader):
    target_label = get_target(labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    total += 1

    # convert an image into adversarial example
    adv_image = zoo_attack(network=model, image=images, t_0=labels)
    adv_image = adv_image.to(device)
    # m
    adv_output = model(adv_image)
    _, adv_pred = adv_output.max(1)
    if adv_pred.item() != labels.item():
        success += 1

    if total >= num_image:
        break

print('success rate : %.4f' % (success / total))
