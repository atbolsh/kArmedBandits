import numpy as np


### The process, changing or not, responsible for the payoffs.
class process:
    """This is the class of the underlying process."""

    def __init__(self, size=10, scale=1.0, update=False, updateScale = 0.01):
        self.choices = np.array([-1, 1])

        self.size = size
        self.update = update
        self.updateScale = updateScale

        self.expected = np.random.randn(size)*scale
        return None
    
    def choice(self, arm):
        v = (self.expected + np.random.randn(self.size))[arm]

        if self.update:
            self.expected += np.random.choice(self.choices, self.size)*self.updateScale
        
        return v

### The top template for agents.
class bandit:
    """This is the agent that tries to play the game. This specific class is supposed to be a template."""

    def __init__(self, size=10):
        self.lastChoice = -1

        self.size = size
        self.count = np.zeros(size)
        self.q = self.initialize()
        return None

    def initialize(self):
        """Use to select initial vals for q. Default is 0"""
        return np.zeros(self.size)
    
    def choose(self):
        """Given all information about prior state, return a value."""
        i = self.index()

        self.count[i] += 1
        self.lastChoice = i

        return i

    def update(self, val):
        """Use self.lastChoice and val to update q """
        return None

    def index(self):
        """Use count and q to select a val"""
        return -1


### Template for all epsilon-greedy classes
class EpsGreedy(bandit):
    """Template for all epsilon-greedy bandits."""
    def __init__(self, size=10, eps=0.0):
        bandit.__init__(self, size)
        
        self.eps = eps
        return None

    def index(self):
        """Use count and q to select a val"""
        if np.random.random() < self.eps:
            return np.random.randint(self.size) 
        else:
            return np.argmax(self.q)


### Template for all UCB classes
class UCB(bandit):
    """Template for all UCB, exploration bandits. See Eq 2.10 in book."""
    def __init__(self, size=10, c=0.0):
        bandit.__init__(self, size)
        
        self.c = c
        return None

    def index(self):
        """Use count and q to select a val. See eq 2.10"""
        step = np.sum(self.count)
        correction = np.sqrt(np.log(step + 0.01)/(self.count + 0.01))
        comparisons = self.q + self.c*correction
        return np.argmax(comparisons)


### Harmonic and gemoetric versions of all of the above.
class HarmonicEps(EpsGreedy):
    def __init__(self, size=10, eps=0.0):
        EpsGreedy.__init__(self, size, eps)      
        return None
   
    def update(self, val):
        """Use self.lastChoice and val to update q """
        self.q[self.lastChoice] += (val - self.q[self.lastChoice])/(self.count[self.lastChoice]) #Has already been updated


class AlphaEps(EpsGreedy):
    def __init__(self, size=10, eps=0.0, alpha=0.05):
        EpsGreedy.__init__(self, size, eps)
        
        self.alpha = alpha
        return None
   
    def update(self, val):
        """Use self.lastChoice and val to update q """
        self.q[self.lastChoice] += (val - self.q[self.lastChoice])*self.alpha


class AlphaEpsUnbiased(EpsGreedy):
    def __init__(self, size=10, eps=0.0, alpha=0.05):
        EpsGreedy.__init__(self, size, eps)
        
        self.alpha = alpha
        self.os = np.zeros(size)
        return None
   
    def update(self, val):
        """Use self.lastChoice and val to update q. See problem 2.7."""
        self.os[self.lastChoice] += self.alpha*(1 - self.os[self.lastChoice])
        beta = self.alpha/self.os[self.lastChoice]
        self.q[self.lastChoice] += (val - self.q[self.lastChoice])*beta


class optimisticAlphaEps(AlphaEps):
    def __init__(self, size=10, eps=0.0, alpha=0.05, optimism=5.0):
        self.o = optimism
        AlphaEps.__init__(self, size, eps, alpha)
        return None
   
    def initialize(self):
        """Use self.lastChoice and val to update q """
        return np.zeros(self.size) + self.o


class optimisticAlphaEpsUbiased(AlphaEps):
    def __init__(self, size=10, eps=0.0, alpha=0.05, optimism=5.0):
        self.o = optimism
        AlphaEpsUnbiased.__init__(self, size, eps, alpha)
        return None
   
    def initialize(self):
        """Use self.lastChoice and val to update q """
        return np.zeros(self.size) + self.o

class optimisticHarmonicEps(HarmonicEps):
    def __init__(self, size=10, eps=0.0, optimism=5.0):
        self.o = optimism
        HarmonicEps.__init__(self, size, eps)
        return None
 
    def initialize(self):
        """Use self.lastChoice and val to update q """
        return np.zeros(self.size) + self.o


class HarmonicUCB(UCB):
    def __init__(self, size=10, c=0.0):
        UCB.__init__(self, size, c)      
        return None
   
    def update(self, val):
        """Use self.lastChoice and val to update q """
        self.q[self.lastChoice] += (val - self.q[self.lastChoice])/(self.count[self.lastChoice]) #Has already been updated


class AlphaUCB(UCB):
    def __init__(self, size=10, c=0.0, alpha=0.05):
        UCB.__init__(self, size, c)
        
        self.alpha = alpha
        return None
   
    def update(self, val):
        """Use self.lastChoice and val to update q """
        self.q[self.lastChoice] += (val - self.q[self.lastChoice])*self.alpha


### The gradient bandit, in a class of its own.
class gradient(bandit):
    """q acts like h here."""
    def __init__(self, size=10, alpha=0.2):
        bandit.__init__(self, size)
        
        self.alpha = alpha
        self.er = 0.0 #Expected reward
        return None
    
    def softmax(self, r):
        e = np.exp(r)
        return e/np.sum(e)

    def sample(self, probs):
        r = np.random.random()
        i = 0
        while r > probs[i]:
            r -= probs[i]
            i += 1
        return i

    def index(self):
        """Use count and q to select a val. See eq 2.10"""
        return self.sample(self.softmax(self.q))

    def update(self, val):
        """See Eq 2.12 """
        d = val - self.er
        self.er += d/np.sum(self.count)
        
        probs = self.softmax(self.q)
        for i in range(self.size):
            if i == self.lastChoice:
                self.q[i] += self.alpha*d*(1 - probs[i])
            else:
                self.q[i] -= self.alpha*d*probs[i]
        return None

