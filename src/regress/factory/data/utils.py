import torch

def cudaify(batch, labels, male=None):
    if type(male) == torch.Tensor:
        return batch.cuda(), labels.cuda(), male.cuda()
    else:
        return batch.cuda(), labels.cuda()