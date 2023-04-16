import numpy as np
import torch
import torch.nn.functional as F


class Attacker:
    def __init__(self, clip_max=0.5, clip_min=-0.5):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        pass


class FGSM(Attacker):
    """
    Fast Gradient Sign Method
    Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy.
    Explaining and Harnessing Adversarial Examples.
    ICLR, 2015
    """

    def __init__(self, eps=0.15, clip_max=0.5, clip_min=-0.5):
        super(FGSM, self).__init__(clip_max, clip_min)
        self.eps = eps

    def generate(self, model, x, y):

        model.eval()

        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)

        nx.requires_grad_()

        out = model(nx)

        loss = F.cross_entropy(out, ny)

        loss.backward()

        x_adv = nx + self.eps * torch.sign(nx.grad.data)

        x_adv.clamp_(self.clip_min, self.clip_max)

        x_adv.squeeze_(0)

        return x_adv.detach()
