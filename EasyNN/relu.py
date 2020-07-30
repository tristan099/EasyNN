"""
class for RelU hidden layer implementation

Paramater Info:
:param der: derivative to carry back during backpropagation
:param LR: Learning rate, effects how fast gradient descent descends the loss function
:param mu: coefficient for momentum
:param epoch: number of iterations for training
:param k: Learning Rate Normalize parameter
:param l1: Amount of L1 Regularization to add
:param l2: Amount of L2 Regularization to add
"""
# lib
import numpy as np

class relu:

    def __init__(self, indims, outdims):
        self.indims = indims
        self.outdims = outdims
        self.w = np.random.randn(self.indims, self.outdims) * \
            np.sqrt(2/(self.indims+self.outdims))
        self.b = np.random.randn(1, self.outdims) * \
            np.sqrt(2/(self.indims+self.outdims))
        self.Gw = np.ones((self.indims, self.outdims))
        self.Gb = np.ones((1, self.outdims))
        self.vw = np.zeros((self.indims, self.outdims))
        self.vb = np.zeros((1, self.outdims))
        self.mw = np.zeros((self.indims, self.outdims))
        self.mb = np.zeros((1, self.outdims))
        self.momw = np.zeros((self.indims, self.outdims))
        self.momb = np.zeros((1, self.outdims))
        self.prevmw = 0
        self.prevmb = 0
        self.LRw = 0
        self.LRb = 0

    def relu(self, x):
        return(x*(x > 0))

    def drelu(self, x):
        return(x > 0)

    def forward(self, a):
        import numpy as np
        self.a = a
        self.g = self.a@self.w
        self.h = self.g+self.b
        self.z = self.relu(self.h)

        return self.z

    def forwarddrop(self, a, p):
        import numpy as np
        self.a = a
        self.M = np.random.rand(self.a.shape[0], self.a.shape[1]) < p
        self.a = (self.a*self.M)/p
        self.g = a@self.w
        self.h = self.g+self.b
        self.z = self.relu(self.h)

        return self.z

    def backward(self, der, LR, mu, epoch, k, l1, l2):
        LR = LR/(k*epoch+1)
        self.der = der
        self.momw = mu*self.momw-LR * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LR * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)
        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der*self.drelu(self.z)@self.w.T

        return (self.der)

    def backwardnesta(self, der, LR, mu, l1, l2):
        self.der = der
        self.momw = mu*self.momw-LR * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LR * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.prevmw = self.momw
        self.prevmb = self.momb

        self.w = self.w+self.prevmw
        self.b = self.b+self.prevmb

    def backwardnestb(self, der, LR, mu, l1, l2):
        gradbw = LR*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LR*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.w = self.w-self.prevmw
        self.b = self.b-self.prevmb

        self.momw = mu*self.prevmw-gradbw
        self.momb = mu*self.prevmb-gradbp

        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der*self.drelu(self.z)@self.w.T
        return (self.der)

    def backwardada(self, der, LR, mu, ep, l1, l2):
        self.der = der

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = LRw*(self.a.T@(self.der*self.drelu(self.z)) +
                      l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der*self.drelu(self.z)) +
                      l1*np.sign(self.b)+l2*self.b)

        self.Gw = self.Gw+gradbw**2
        self.Gb = self.Gb+gradbp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der*self.drelu(self.z))+l1*np.sign(self.b)+l2*self.b)

        prevmw = self.momw
        prevmb = self.momb

        self.w = self.w+prevmw
        self.b = self.b+prevmb

        self.der = self.der*self.drelu(self.z)@self.w.T
        return (self.der)

    def backwardadanesta(self, der, LR, mu, ep, l1, l2):
        self.der = der
        self.LRw = LR/np.sqrt(self.Gw+ep)
        self.LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb*(np.sum(self.der*self.drelu(self.z)
                                  )+l1*np.sign(self.b)+l2*self.b)

        self.Gw = self.Gw+gradbw**2
        self.Gb = self.Gb+gradbp**2

        self.momw = mu*self.momw-self.LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-self.LRb * \
            (np.sum(self.der*self.drelu(self.z))+l1*np.sign(self.b)+l2*self.b)

        self.prevmw = self.momw
        self.prevmb = self.momb

        self.w = self.w+self.prevmw
        self.b = self.b+self.prevmb

    def backwardadanestb(self, der, LR, mu, ep, l1, l2):
        self.der = der
        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb*(np.sum(self.der*self.drelu(self.z)
                                  )+l1*np.sign(self.b)+l2*self.b)

        self.w = self.w-self.prevmw
        self.b = self.b-self.prevmb

        self.momw = mu*self.prevmw-gradbw
        self.momb = mu*self.prevmb-gradbp

        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der*self.drelu(self.z)@self.w.T
        return(self.der)

    def backwardrms(self, der, LR, mu, ep, gam, l1, l2):
        self.der = der

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = LRw*(self.a.T@(self.der*self.drelu(self.z)) +
                      l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der*self.drelu(self.z)) +
                      l1*np.sign(self.b)+l2*self.b)

        self.Gw = gam*self.Gw+(1-gam)*gradbw**2
        self.Gb = gam*self.Gb+(1-gam)*gradbp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der*self.drelu(self.z))+l1*np.sign(self.b)+l2*self.b)

        prevmw = self.momw
        prevmb = self.momb

        self.w = self.w+prevmw
        self.b = self.b+prevmb

        self.der = self.der*self.drelu(self.z)@self.w.T
        return (self.der)

    def backwardrmsnesta(self, der, LR, mu, ep, gam, l1, l2):
        self.der = der
        self.LRw = LR/np.sqrt(self.Gw+ep)
        self.LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb*(np.sum(self.der*self.drelu(self.z)
                                  )+l1*np.sign(self.b)+l2*self.b)

        self.Gw = gam*self.Gw+(1-gam)*gradbw**2
        self.Gb = gam*self.Gb+(1-gam)*gradbp**2

        self.momw = mu*self.momw-self.LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-self.LRb * \
            (np.sum(self.der*self.drelu(self.z))+l1*np.sign(self.b)+l2*self.b)

        self.prevmw = self.momw
        self.prevmb = self.momb

        self.w = self.w+self.prevmw
        self.b = self.b+self.prevmb

    def backwardrmsnestb(self, der, LR, mu, ep, l1, l2):
        self.der = der
        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelu(self.z))+l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb*(np.sum(self.der*self.drelu(self.z)
                                  )+l1*np.sign(self.b)+l2*self.b)

        self.w = self.w-self.prevmw
        self.b = self.b-self.prevmb

        self.momw = mu*self.prevmw-gradbw
        self.momb = mu*self.prevmb-gradbp

        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der*self.drelu(self.z)@self.w.T
        return(self.der)

    def backwardadam(self, der, LR, mu, ep, gam, epoch, l1, l2):
        self.der = der

        self.mw = mu*self.mw+(1-mu)*(self.a.T@self.der +
                                     l1*np.sign(self.w)+l2*self.w)
        self.vw = gam*self.vw+(1-gam)*(self.a.T@self.der +
                                       l1*np.sign(self.b)+l2*self.b)**2
        self.mhat = self.mw/(1+mu**epoch)
        self.vhat = self.vw/(1+gam**epoch)

        self.mb = mu*self.mb+(1-mu)*(np.sum(self.der) +
                                     l1*np.sign(self.w)+l2*self.w)
        self.vb = gam*self.vb+(1-gam)*(np.sum(self.der) +
                                       l1*np.sign(self.b)+l2*self.b)**2
        self.mhatb = self.mb/(1+mu**epoch)
        self.vhatb = self.vb/(1+gam**epoch)

        LRw = LR/np.sqrt(self.vhat+ep)
        LRb = LR/np.sqrt(self.vhatb+ep)

        self.w = self.w-LRw*self.mhat
        self.b = self.b-LRb*(np.sum(self.mhatb))

        self.der = self.der*self.drelu(self.z)@self.w.T
        return(self.der)
