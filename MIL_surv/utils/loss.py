import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def define_loss(args):
    if args.loss == "ce_surv":
        loss = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.loss == "nll_surv":
        loss = NLLSurvLoss(delta_alpha = args.delta_alpha, alpha=args.alpha_surv)
    elif args.loss == "nll_surv_l1":
        loss = [NLLSurvLoss(alpha=args.alpha_surv), nn.L1Loss()]
    elif args.loss == "nll_surv_mse":
        loss = [NLLSurvLoss(alpha=args.alpha_surv), nn.MSELoss()]
    elif args.loss == "nll_surv_kl":
        loss = [NLLSurvLoss(alpha=args.alpha_surv), KLLoss()]
    elif args.loss == "nll_surv_cos":
        loss = [NLLSurvLoss(alpha=args.alpha_surv), CosineLoss()]
    else:
        raise NotImplementedError
    return loss


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

# def delta_loss(delta, delta_pred):
#     batch_size = len(delta)
#     delta = delta.view(batch_size, 1)  
#     delta_pred = delta_pred.view(batch_size, 1) #.float()  # censorship status, 0 or 1
  
#     #loss = torch.log(1 + torch.exp(-(2*delta - 1)*(2*delta_pred - 1)))
#     loss = torch.log(1 + torch.exp(-(2*delta - 1)*delta_pred))
#     loss = loss.mean()
#     return loss

def delta_l2_loss(delta, delta_pred):
    batch_size = len(delta)
    delta = delta.view(batch_size, 1)  
    delta_pred = delta_pred.view(batch_size, 1) #.float()  # censorship status, 0 or 1
    loss = (delta - delta_pred).pow(2).sum(0).sqrt()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, delta_alpha, alpha):
        self.alpha = alpha
        self.delta_alpha = delta_alpha
        print(self.alpha, self.delta_alpha)
        
    def __call__(self, hazards, S, Y, c,delta, delta_pred):
        return (1-self.delta_alpha)*nll_loss(hazards, S, Y, c, alpha=self.alpha) + self.delta_alpha*delta_l2_loss(delta, delta_pred)


class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")


class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)
