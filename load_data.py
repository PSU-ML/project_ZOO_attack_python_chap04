
import torchvision
import torchvision.transforms as transforms

def load_data(data='cifar10'):
    if data=='cifar10':
        return load_cifar10()
    if data=='mnist':
        return load_mnist()



def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return training_set, test_set

def load_mnist():
    '''
    the mnist dataset is resized to 32 * 32
    :return:
    '''
    transform = transforms.Compose([

        transforms.Resize(32),
        transforms.ToTensor()
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return training_set, test_set


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

    class UnNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            """
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:
                Tensor: Normalized image.
            """
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return tensor

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
