import torch

def PairwiseConfusion(features):
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')
    batch_left = features[:int(0.5*batch_size)]
    batch_right = features[int(0.5*batch_size):]
    loss  = torch.norm((batch_left - batch_right).abs(),2, 1).sum() / float(batch_size)

    return loss

def EntropicConfusion(features):
    batch_size = features.size(0)
    return torch.mul(features, torch.log(features)).sum() * (1.0 / batch_size)