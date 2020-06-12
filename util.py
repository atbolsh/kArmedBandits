from classes import *

def pull(p, b):
    """All the methods that need to be called for one lever pull."""
    i = b.choose()
    v = p.choice(i)
    b.update(v)
    return i, v

def run(p, btype, size=1000):
    """A full run. Return np array of sequences."""
    result = np.zeros(size)
    b = btype()
    for i in range(size):
        _, v = pull(p, b)
        result[i] = v
    return result

def means(btype, update=False, size=1000, runs=1000):
    m = np.zeros(size)
    for i in range(runs):
        p = process(update=update)
        r = run(p, btype, size)
        m += (r - m)/(i + 1)
    return m


    
