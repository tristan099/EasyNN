# lib
import numpy as np

class prelu:

    def __init__(self, indims, outdims):
        self.indims = indims
        self.outdims = outdims
        self.w = np.random.randn(self.indims, self.outdims) * \
            np.sqrt(2/(self.indims+self.outdims))
        self.b = np.random.randn(1, self.outdims) * \
            np.sqrt(2/(self.indims+self.outdims))
        self.Gw = np.ones((self.indims, self.outdims))
        self.Gb = np.ones((1, self.outdims))
        self.Gp = np.ones((self.outdims))
        self.vw = np.zeros((self.indims, self.outdims))
        self.vb = np.zeros((1, self.outdims))
        self.vp = np.ones((1, self.outdims))
        self.mw = np.zeros((self.indims, self.outdims))
        self.mb = np.zeros((1, self.outdims))
        self.mp = np.ones((1, self.outdims))
        self.momw = np.zeros((self.indims, self.outdims))
        self.momb = np.zeros((1, self.outdims))
        self.momp = np.ones((1, self.outdims))
        self.prevmw = 0
        self.prevmb = 0
        self.LRw = 0
        self.LRb = 0
        self.LRp = 0
        self.p = np.ones((1, self.outdims))

    def prelu(self, x, p):
        return(x*(x > 0)+p*(x <= 0))

    def drelup(self, x):
        return((x <= 0))

    def drelux(self, x, p):
        return((x > 0)+p*(x <= 0))

    def forward(self, a):
        self.a = a
        self.g = a@self.w
        self.h = self.g+self.b
        self.z = self.prelu(self.h, self.p)
        return self.z

    def forwarddrop(self, a, p):
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
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LR * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        self.momp = mu*self.momp-LR * \
            (np.sum(self.der*self.drelup(self.z))+l1*np.sign(self.p)+l2*self.p)
        self.w = self.w+self.momw
        self.b = self.b+self.momb
        self.p = self.p+self.momp

        self.der = self.der*self.drelux(self.z, self.p)@self.w.T

        return (self.der)

    def backwardnesta(self, der, LR, mu, l1, l2):
        self.der = der
        self.momw = mu*self.momw-LR * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LR * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        self.momp = mu*self.momp-LR * \
            (np.sum(self.der*self.drelup(self.z))+l1*np.sign(self.p)+l2*self.p)
        self.prevmw = self.momw
        self.prevmb = self.momb
        self.prevmp = self.momp

        self.w = self.w+self.prevmw
        self.b = self.b+self.prevmb
        self.p = self.p+self.prevmp

        self.forward(self.a)

    def backwardnestb(self, der, LR, mu, l1, l2):

        gradbw = LR*(self.a.T@(self.der*self.drelux(self.z, self.p)
                               )+l1*np.sign(self.w)+l2*self.w)
        gradbp = LR*(np.sum(self.der*self.drelux(self.z, self.p)) +
                     l1*np.sign(self.b)+l2*self.b)
        gradp = LR*(np.sum(self.der*self.drelup(self.z)) +
                    l1*np.sign(self.p)+l2*self.p)
        self.w = self.w-self.prevmw
        self.b = self.b-self.prevmb
        self.p = self.p-self.prevmp

        self.momw = mu*self.prevmw-gradbw
        self.momb = mu*self.prevmb-gradbp
        self.momp = mu*self.prevmp-gradp

        self.w = self.w+self.momw
        self.b = self.b+self.momb
        self.p = self.p+self.momp

        self.der = self.der*self.drelux(self.z, self.p)@self.w.T
        return (self.der)

    def backwardada(self, der, LR, mu, ep, l1, l2):

        self.der = der

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)
        LRp = LR/np.sqrt(self.Gp+ep)

        gradbw = LRw*(self.a.T@(self.der*self.drelux(self.z,
                                                     self.p))+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der*self.drelux(self.z, self.p)
                             )+l1*np.sign(self.b)+l2*self.b)
        gradp = LR*(np.sum(self.der*self.drelup(self.z)) +
                    l1*np.sign(self.p)+l2*self.p)
        self.Gw = self.Gw+gradbw**2
        self.Gb = self.Gb+gradbp**2
        self.Gp = self.Gp+gradp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        self.momp = mu*self.momp-LRp * \
            (np.sum(self.der*self.drelup(self.z))+l1*np.sign(self.p)+l2*self.p)

        prevmw = self.momw
        prevmb = self.momb
        prevmp = self.momp

        self.w = self.w+prevmw
        self.b = self.b+prevmb
        self.p = self.p+prevmp

        self.der = self.der*self.drelux(self.z, self.p)@self.w.T
        return (self.der)

    def backwardadanesta(self, der, LR, mu, ep, l1, l2):
        self.der = der
        self.LRw = LR/np.sqrt(self.Gw+ep)
        self.LRb = LR/np.sqrt(self.Gb+ep)
        self.LRp = LR/np.sqrt(self.Gp+ep)

        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        gradp = self.LRp*(np.sum(self.der*self.drelup(self.z)
                                 )+l1*np.sign(self.p)+l2*self.p)

        self.Gw = self.Gw+gradbw**2
        self.Gb = self.Gb+gradbp**2
        self.Gp = self.Gp+gradp**2

        self.momw = mu*self.momw-self.LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-self.LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        self.momp = mu*self.momp-self.LRp * \
            (np.sum(self.der*self.drelup(self.z))+l1*np.sign(self.p)+l2*self.p)
        self.prevmw = self.momw
        self.prevmb = self.momb
        self.prevmp = self.momp

        self.w = self.w+self.prevmw
        self.b = self.b+self.prevmb
        self.p = self.p+self.prevmp

        self.forward(self.a)

    def backwardadanestb(self, der, LR, mu, ep, l1, l2):

        self.der = der
        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        gradp = self.LRp*(np.sum(self.der*self.drelup(self.z)
                                 )+l1*np.sign(self.p)+l2*self.p)

        self.w = self.w-self.prevmw
        self.b = self.b-self.prevmb
        self.p = self.p-self.prevmp

        self.momw = mu*self.prevmw-gradbw
        self.momb = mu*self.prevmb-gradbp
        self.momp = mu*self.prevmp-gradp

        self.w = self.w+self.momw
        self.b = self.b+self.momb
        self.p = self.p+self.momp

        self.der = self.der*self.drelux(self.z, self.p)@self.w.T
        return(self.der)

    def backwardrms(self, der, LR, mu, ep, gam, l1, l2):

        self.der = der

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)
        LRp = LR/np.sqrt(self.Gp+ep)

        gradbw = LRw*(self.a.T@(self.der*self.drelux(self.z,
                                                     self.p))+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der*self.drelux(self.z, self.p)
                             )+l1*np.sign(self.b)+l2*self.b)
        gradp = LRp*(np.sum(self.der*self.drelup(self.z)) +
                     l1*np.sign(self.p)+l2*self.p)

        self.Gw = gam*self.Gw+(1-gam)*gradbw**2
        self.Gb = gam*self.Gb+(1-gam)*gradbp**2
        self.Gp = gam*self.Gp+(1-gam)*gradp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        self.momp = mu*self.momp-LRp * \
            (np.sum(self.der*self.drelup(self.z))+l1*np.sign(self.p)+l2*self.p)

        prevmw = self.momw
        prevmb = self.momb
        prevmp = self.momp

        self.w = self.w+prevmw
        self.b = self.b+prevmb
        self.p = self.p+prevmp

        self.der = self.der*self.drelux(self.z, self.p)@self.w.T
        return (self.der)

    def backwardrmsnesta(self, der, LR, mu, ep, gam, l1, l2):
        self.der = der
        self.LRw = LR/np.sqrt(self.Gw+ep)
        self.LRb = LR/np.sqrt(self.Gb+ep)
        self.LRp = LR/np.sqrt(self.Gp+ep)

        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        gradp = self.LRp*(np.sum(self.der*self.drelup(self.z)
                                 )+l1*np.sign(self.p)+l2*self.p)

        self.Gw = gam*self.Gw+(1-gam)*gradbw**2
        self.Gb = gam*self.Gb+(1-gam)*gradbp**2
        self.Gp = gam*self.Gp+(1-gam)*gradp**2

        self.momw = mu*self.momw-self.LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-self.LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        self.momp = mu*self.momp-self.LRp * \
            (np.sum(self.der*self.drelup(self.z))+l1*np.sign(self.p)+l2*self.p)

        self.prevmw = self.momw
        self.prevmb = self.momb
        self.prevmp = self.momp

        self.w = self.w+self.prevmw
        self.b = self.b+self.prevmb
        self.p = self.p+self.prevmp

        self.forward(self.a)

    def backwardrmsnestb(self, der, LR, mu, ep, l1, l2):
        self.der = der
        gradbw = self.LRw * \
            (self.a.T@(self.der*self.drelux(self.z, self.p)) +
             l1*np.sign(self.w)+l2*self.w)
        gradbp = self.LRb * \
            (np.sum(self.der*self.drelux(self.z, self.p))+l1*np.sign(self.b)+l2*self.b)
        gradp = self.LRp*(np.sum(self.der*self.drelup(self.z)
                                 )+l1*np.sign(self.p)+l2*self.p)

        self.w = self.w-self.prevmw
        self.b = self.b-self.prevmb
        self.p = self.p-self.prevmp

        self.momw = mu*self.prevmw-gradbw
        self.momb = mu*self.prevmb-gradbp
        self.momp = mu*self.prevmp-gradp

        self.w = self.w+self.momw
        self.b = self.b+self.momb
        self.p = self.p+self.momp

        self.der = self.der*self.drelux(self.z, self.p)@self.w.T
        return(self.der)

    def backwardadam(self, der, LR, mu, ep, gam, epoch, l1, l2):
        self.der = der

        self.mw = mu*self.mw + \
            (1-mu)*(self.a.T@(self.der*self.drelux(self.z, self.p)) +
                    l1*np.sign(self.w)+l2*self.w)
        self.vw = gam*self.vw + \
            (1-gam)*(self.a.T@(self.der*self.drelux(self.z, self.p)) +
                     l1*np.sign(self.w)+l2*self.w)**2
        self.mhat = self.mw/(1+mu**epoch)
        self.vhat = self.vw/(1+gam**epoch)

        self.mb = mu*self.mb + \
            (1-mu)*(np.sum(self.der*self.drelux(self.z, self.p)) +
                    l1*np.sign(self.b)+l2*self.b)
        self.vb = gam*self.vb + \
            (1-gam)*(np.sum(self.der*self.drelux(self.z, self.p)) +
                     l1*np.sign(self.b)+l2*self.b)**2
        self.mhatb = self.mb/(1+mu**epoch)
        self.vhatb = self.vb/(1+gam**epoch)

        self.mp = mu*self.mp + \
            (1-mu)*(np.sum(self.der*self.drelup(self.z))+l1*np.sign(self.p)+l2*self.p)
        self.vp = gam*self.vp + \
            (1-gam)*(np.sum(self.der*self.drelup(self.z)) +
                     l1*np.sign(self.p)+l2*self.p)**2
        self.mhatp = self.mp/(1+mu**epoch)
        self.vhatp = self.vp/(1+gam**epoch)

        LRw = LR/np.sqrt(self.vhat+ep)
        LRb = LR/np.sqrt(self.vhatb+ep)
        LRp = LR/np.sqrt(self.vhatp+ep)

        self.w = self.w-LRw*self.mhat
        self.b = self.b-LRb*(np.sum(self.mhatb))
        self.p = self.p-LRp*(np.sum(self.mhatp))

        self.der = self.der*self.drelux(self.z, self.p)@self.w.T
        return(self.der)
