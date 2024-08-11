from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from geomloss import SamplesLoss

class SinkhornDistance(torch.nn.Module):
    r"""
    https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Pytorch_Wasserstein.ipynb
    
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        self.actual_nits = actual_nits
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

class mIOS(nn.Module):
    def __init__(self, n_classes=5, eps=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.eps = nn.parameter.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.ios_class = nn.parameter.Parameter(data=torch.zeros(self.n_classes), requires_grad=False)
    
    def forward(self, pred, labels):
        pred = torch.argmax(pred, dim=1)
        for class_ in range(self.n_classes):
            pred_idx = (pred == class_)
            labels_idx = (labels == class_)
            inter = torch.logical_and(pred_idx, labels_idx).sum()
            union = torch.logical_or(pred_idx, labels_idx).sum()
            self.ios_class[class_] += (inter / (union + self.eps)).mean()
        return self.ios_class.mean()
    

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=5, eps=1e-5, weights=None, reduce=True):
        super().__init__()
        self.reduce = reduce
        self.n_classes = n_classes
        self.eps = nn.parameter.Parameter(data=torch.tensor(eps), requires_grad=False)
        if weights is None:
            self.weights = nn.parameter.Parameter(data=torch.ones(self.n_classes), requires_grad=False)
        else:
            self.weights = nn.parameter.Parameter(data=torch.tensor(weights), requires_grad=False)
  
    def forward(self, pred, labels):
        loss = 0
        for class_ in range(self.n_classes):
            pred = nn.functional.softmax(pred, dim=1)
            pred_probs = pred[:, class_]
            labels_idx = (labels == class_)
            inter = (pred_probs * labels_idx).sum()
            delimiter = pred_probs.sum() + labels_idx.sum()
            loss += -self.weights[class_] * (inter / (delimiter + self.eps))
        if self.reduce:
            loss = loss.mean()
        return loss


class SoftDiceCrossLoss(nn.Module):
    def __init__(self, n_classes=5, eps=1e-5, class_weights=None, loss_weights=(0.5, 0.5)):
        super().__init__()
        self.soft_dice_loss = SoftDiceLoss(n_classes, eps, class_weights)
        self.cross_loss = torch.nn.CrossEntropyLoss(class_weights)
        self.loss_weights = loss_weights

    def forward(self, pred, labels):
        return self.loss_weights[0] * self.soft_dice_loss(pred, labels) +\
               self.loss_weights[1] * self.cross_loss(pred, labels)

class SoftDiceCrossWeightedLoss(nn.Module):
    def __init__(self, n_classes=5, eps=1e-5, class_weights=None, loss_weights=(0.5, 0.5)):
        super().__init__()
        self.soft_dice_loss = SoftDiceLoss(n_classes, eps, class_weights, reduce=False)
        self.cross_loss = torch.nn.CrossEntropyLoss(class_weights, reduction="none")
        self.loss_weights = loss_weights

    def forward(self, pred, labels, probs):
        loss_cross = self.loss_weights[1] * self.cross_loss(pred, labels).mean(dim=(1,2))
        loss_soft_dice = self.loss_weights[0] * self.soft_dice_loss(pred, labels)
        loss =  loss_soft_dice + loss_cross
        return (probs * loss).mean()


def double_flatten(x, y):
    x = torch.flatten(x, start_dim=1)
    y = torch.flatten(y, start_dim=1)
    return x, y

class DistanceABS():
    def __call__(self, x, y):
        x, y = double_flatten(x, y)
        x = x.mean()
        y = y.mean()
        return torch.abs(x - y)

class DistanceSinkhorn():
    def __init__(self):
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    def __call__(self, x, y):
        x, y = double_flatten(x, y)
        loss = self.loss(x, y)
        return loss


class MathcingMeanLoss(nn.Module):
    def __init__(self, lambdas, distance):
        super().__init__()
        self.lambdas = nn.parameter.Parameter(data=torch.tensor(lambdas), requires_grad=False)
        self.len_lambdas = len(self.lambdas)
        self.means = nn.parameter.Parameter(data=torch.zeros(len(lambdas)), requires_grad=False)
        self.distance = distance
        
    def forward(self, domain_1, domain_2):
        tot_loss = 0
        for i in range(len(domain_1)):
            tot_loss += self.lambdas[i] * self.distance(domain_1[i], domain_2[i])
        return tot_loss


class SegMathcingLoss(nn.Module):
    def __init__(self, lambdas, distance=DistanceABS(), n_classes=5, eps=1e-5, weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.seg_loss = SoftDiceCrossLoss()
        if weights is None:
            self.weights = nn.parameter.Parameter(data=torch.ones(self.n_classes), requires_grad=False)
        else:
            self.weights = nn.parameter.Parameter(data=torch.tensor(weights), requires_grad=False)
        self.match_loss = MathcingMeanLoss(lambdas, distance)
  
    def forward(self, source_enc, source_mask, source_pred, target_enc):
        mathc_loss = self.match_loss(source_enc, target_enc)
        seg_loss = self.seg_loss(source_pred, source_mask)
        return seg_loss + mathc_loss







