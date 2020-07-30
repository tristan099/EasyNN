# lib
import numpy as np

class softmax:

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

    def softmax(self, mat):
        nom = np.exp(mat)
        denom = np.exp(mat).sum(1).reshape(nom.shape[0], 1)
        return(nom/denom)

    def forward(self, a):
        self.a = a
        self.g = a@self.w
        self.h = self.g+self.b
        self.z = self.softmax(self.h)

        return self.z

    def forwarddrop(self, a, p):
        self.a = a
        self.M = np.random.rand(self.a.shape[0], self.a.shape[1]) < p
        self.a = (self.a*self.M)/p
        self.g = a@self.w
        self.h = self.g+self.b
        self.z = self.relu(self.h)

        return self.z

    def backward(self, LR, y, mu, epoch, k, l1, l2):
        LR = LR/(k*epoch+1)
        self.der = self.z-y
        self.momw = mu*self.momw-LR * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LR * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)
        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der@self.w.T

        return (self.der)

    def backwardnest(self, LR, y, mu, l1, l2):
        self.der = self.z-y
        self.momw = mu*self.momw-LR * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LR * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        prevmw = self.momw
        prevmb = self.momb

        self.w = self.w+prevmw
        self.b = self.b+prevmb

        self.forward(self.a)
        self.der = self.z-y

        gradbw = LR*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LR*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.w = self.w-prevmw
        self.b = self.b-prevmb

        self.momw = mu*prevmw-gradbw
        self.momb = mu*prevmb-gradbp

        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der@self.w.T
        return (self.der)

    def backwardada(self, LR, mu, ep, y, l1, l2):
        self.der = self.z-y

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = LRw*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.Gw = self.Gw+gradbw**2
        self.Gb = self.Gb+gradbp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        prevmw = self.momw
        prevmb = self.momb

        self.w = self.w+prevmw
        self.b = self.b+prevmb

        self.der = self.der@self.w.T

        return (self.der)

    def backwardadanest(self, LR, mu, ep, y, l1, l2):
        self.der = self.z-y

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = LRw*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.Gw = self.Gw+gradbw**2
        self.Gb = self.Gb+gradbp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        prevmw = self.momw
        prevmb = self.momb

        self.w = self.w+prevmw
        self.b = self.b+prevmb

        self.forward(self.a)
        self.der = self.z-y

        gradbw = LRw*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.w = self.w-prevmw
        self.b = self.b-prevmb

        self.momw = mu*prevmw-gradbw
        self.momb = mu*prevmb-gradbp

        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der@self.w.T
        return(self.der)

    def backwardrms(self, LR, mu, ep, y, gam, l1, l2):
        self.der = self.z-y

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = LRw*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.Gw = gam*self.Gw+(1-gam)*gradbw**2
        self.Gb = gam*self.Gb+(1-gam)*gradbp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        prevmw = self.momw
        prevmb = self.momb

        self.w = self.w+prevmw
        self.b = self.b+prevmb

        self.der = self.der@self.w.T

        return (self.der)

    def backwardrmsnest(self, LR, mu, ep, y, gam, l1, l2):
        self.der = self.z-y

        LRw = LR/np.sqrt(self.Gw+ep)
        LRb = LR/np.sqrt(self.Gb+ep)

        gradbw = LRw*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.Gw = gam*self.Gw+(1-gam)*gradbw**2
        self.Gb = gam*self.Gb+(1-gam)*gradbp**2

        self.momw = mu*self.momw-LRw * \
            (self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        self.momb = mu*self.momb-LRb * \
            (np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        prevmw = self.momw
        prevmb = self.momb

        self.w = self.w+prevmw
        self.b = self.b+prevmb

        self.forward(self.a)
        self.der = self.z-y

        gradbw = LRw*(self.a.T@(self.der)+l1*np.sign(self.w)+l2*self.w)
        gradbp = LRb*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)

        self.w = self.w-prevmw
        self.b = self.b-prevmb

        self.momw = mu*prevmw-gradbw
        self.momb = mu*prevmb-gradbp

        self.w = self.w+self.momw
        self.b = self.b+self.momb

        self.der = self.der@self.w.T
        return(self.der)

    def backwardadam(self, LR, mu, ep, gam, y, epoch, l1, l2):
        self.der = self.z-y

        self.mw = mu*self.mw+(1-mu)*(self.a.T@self.der +
                                     l1*np.sign(self.w)+l2*self.w)
        self.vw = gam*self.vw+(1-gam)*(self.a.T@self.der +
                                       l1*np.sign(self.w)+l2*self.w)**2
        self.mhat = self.mw/(1+mu**epoch)
        self.vhat = self.vw/(1+gam**epoch)

        self.mb = mu*self.mb+(1-mu)*LR*np.sum(self.der +
                                              l1*np.sign(self.b)+l2*self.b)
        self.vb = gam*self.vb + \
            (1-gam)*LR*(np.sum(self.der)+l1*np.sign(self.b)+l2*self.b)**2
        self.mhatb = self.mb/(1+mu**epoch)
        self.vhatb = self.vb/(1+gam**epoch)

        LRw = LR/np.sqrt(self.vhat+ep)
        LRb = LR/np.sqrt(self.vhatb+ep)

        self.w = self.w-LRw*self.mhat
        self.b = self.b-LRb*(np.sum(self.mhatb))

        self.der = self.der@self.w.T

        return(self.der)
