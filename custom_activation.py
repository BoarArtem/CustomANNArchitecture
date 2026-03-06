from torch import nn

class SqeLon(nn.Module):
    def __init__(self, epsilon = 0.01):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, z):
        return 1 / ((z - 1)**2 + self.epsilon)

# activation = CustomActivation()
#
# a1 = int(input("a1: "))
# w4 = int(input("w4: "))
# b2 = int(input("b2: "))
#
# z2 = a1 * w4 + b2
# a2 = activation.forward_activation(z2)
# print(a2)