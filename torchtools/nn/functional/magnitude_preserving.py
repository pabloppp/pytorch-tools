import torch

def mp_cat(*args, dim=1, t=0.5):
    if isinstance(t, float):
        t = [1-t, t]
    assert len(args) == len(t), "t must be a single scalar or a list of scalars of length len(args)"
    
    w = [m/a.size(dim)**0.5 for a, m in zip(args, t)]
    C = (sum([a.size(dim) for a in args]) / sum([m**2 for m in t]))**0.5

    return torch.cat([a*v for a, v in zip(args, w)], dim=dim) * C

def mp_sum(*args, t=0.5):
    if isinstance(t, float):
        t = [1-t, t]

    assert len(args) == len(t), "t must be a single scalar or a list of scalars of length len(args)"
    assert abs(sum(t)-1) < 1e-3 , "the values of t should all add up to one"

    return sum([a*m for a, m in zip(args, t)]) / sum([m**2 for m in t])**0.5
