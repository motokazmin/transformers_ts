import torch

def KrvCrossEntropyLoss(logits, target_indexes, tok, device, reduction = 'mean'):
    not_mask_indices = (target_indexes != -100).nonzero(as_tuple=True)[0]
    logits = logits[not_mask_indices, :]
    target_indexes = target_indexes[not_mask_indices]

    loss = -torch.log(logits[range(len(target_indexes)), target_indexes].exp()/logits.exp().sum(-1))
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss

def KrvCrossEntropyLossEx(logits, target_indexes, tok, device, reduction = 'mean'):
    not_mask_indices = (target_indexes != -100).nonzero(as_tuple=True)[0]
    logits = logits[not_mask_indices, :]
    target_indexes = target_indexes[not_mask_indices]
    tokens = tok.convert_ids_to_tokens(target_indexes)
    
    target_vals = tok.convert_tokens_to_vals(tokens)
    delta = tok.get_valid_interval()
    for i, l in enumerate(logits):
        logit = logits[i, :]
        tv = torch.tensor([t if t > -10000 else target_vals[i] for t in tok.values]).to(device)
        
        logits[i, :] = logits[i, :] *(1 - abs(target_vals[i] - tv)/delta)
    
    loss = -torch.log(logits[range(len(target_indexes)), target_indexes].exp()/logits.exp().sum(-1))
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss
