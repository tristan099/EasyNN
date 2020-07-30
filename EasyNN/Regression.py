"""
Simple regression Class to utilize optimizers for Linear/Logistic Regression Problems
"""
# lib
import numpy as np

class Regression:

    def __init__(self, activation=None, outdims=0, addbasis=False, basisfunc=None):
        self.activation = activation
        self.outdims = outdims
        self.basisfunc = basisfunc
        self.addbasis = addbasis

    def wi(self, x):
        self.ones = np.ones((x.shape[0], 1))
        if self.addbasis == True:
            self.phi = np.hstack((self.ones, x, self.basisfunc))
        else:
            self.phi = np.hstack((self.ones, x))
        self.G = np.ones((self.phi.shape[1], self.outdims))
        self.w = np.random.randn(self.phi.shape[1], self.outdims)
        self.mom = np.random.randn(self.phi.shape[1], self.outdims)
        self.mw = np.zeros((self.phi.shape[1], self.outdims))
        self.vw = np.zeros((self.phi.shape[1], self.outdims))

    def train(self, x=None, y=None, LR=1e-4, L1=0, L2=0, epochs=1000, multi=False, lin=True, steps=None):
        err = []
        if not steps:
            steps = x.shape[0]
        if lin:
            for i in range(epochs):
                for a in range(0, x.shape[0], steps):
                    self.w = self.w-LR*(self.phi[a:(a+steps), :].T@(
                        self.phi[a:(a+steps), :]@self.w-y[a:(a+steps), :])-L1*np.sign(self.w)-L2*self.w)
                    self.predictions = self.phi[a:(a+steps), :]@self.w
            if multi:
                self.errs = np.trace(
                    (self.predictions-y[a:(a+steps), :]).T@(self.predictions-y[a:(a+steps), :]))
                return("Error-", self.errs)
            else:
                self.errs = (
                    self.predictions-y[a:(a+steps), :]).T@(self.predictions-y[a:(a+steps), :])
            return("Error-", self.errs)

        else:
            for i in range(epochs):
                for a in range(0, e.shape[0], steps):
                    self.mom = self.mom-LR * \
                        self.phi[a:(a+steps), :].T@(self.p -
                                                    y[a:(a+steps), :])-L1*np.sign(self.w)-L2*self.w
                    self.p = self.activation(self.phi[a:(a+steps), :]@self.w)
                    self.w = self.w+self.mom
                if multi == True:
                    self.errs = -np.sum(y[a:(a+steps), :]*np.log(self.p))
                    self.err = err.append(err, errs)
                else:
                    self.errs = - \
                        np.sum(y[a:(a+steps), :]*np.log(self.p) +
                               (1-y[a:(a+steps), :])*np.log(1-self.p))
                    self.predictions = np.rint(self.p)
            plt.plot(err)
            plt.show()
            return(print("cost-", self.errs))

    def trainada(self, x=None, y=None, LR=1e-4, L1=0, L2=0, epochs=1000, multi=False, ep=1e-6, lin=True, steps=None):
        if not steps:
            steps = x.shape[0]
        if lin:
            for i in range(epochs):
                for a in range(0, x.shape[0], steps):
                    self.p = (self.phi[a:(a+steps), :]@self.w)

                    self.grad = (self.phi[a:(a+steps), :].T)@(self.p -
                                                              y[a:(a+steps), :])-L1*np.sign(self.w)-L2*self.w
                    self.G = self.G+self.grad**2
                    LRe = LR/np.sqrt(self.G+ep)

                    self.mom = mu*self.mom-LRe * \
                        ((self.phi[a:(a+steps), :].T@(self.p -
                                                      y[a:(a+steps), :]))-L1*np.sign(self.w)-L2*self.w)

                    self.w = self.w+self.mom

            if multi:
                self.errs = np.trace(
                    (self.p-y[a:(a+steps), :]).T@(self.p-y[a:(a+steps), :]))
                return("Error-", self.errs)
            else:
                self.errs = (self.p-y[a:(a+steps), :]
                             ).T@(self.p-y[a:(a+steps), :])
                return("Error-", self.errs)
        else:
            for i in range(epochs):
                for a in range(0, x.shape[0], steps):
                    self.p = self.activation(self.phi[a:(a+steps), :]@self.w)

                    self.grad = (self.phi[a:(a+steps), :].T)@(self.p -
                                                              y[a:(a+steps), :])-L1*np.sign(self.w)-L2*self.w
                    self.G = self.G+self.grad**2
                    LRe = LR/np.sqrt(self.G+ep)

                    self.mom = mu*self.mom-LRe * \
                        ((self.phi[a:(a+steps), :].T@(self.p -
                                                      y[a:(a+steps), :]))-L1*np.sign(self.w)-L2*self.w)

                    self.w = self.w+self.mom
                    self.pred = np.rint(self.p)
                    self.acc = np.mean(self.pred == y[a:(a+steps), :])

                if self.acc == 1.0:
                    print(i)
                    break
            if multi:
                self.errs = -np.sum(y[a:(a+steps), :]*np.log(self.p))
            else:
                self.errs = - \
                    np.sum(y[a:(a+steps), :]*np.log(self.p) +
                           (1-y[a:(a+steps), :])*np.log(1-self.p))
            self.predictions = np.rint(self.p)
            print(np.mean(mod.predictions == y[a:(a+steps), :]))
            return(print("cost-", self.errs))

    def trainrms(self, x=None, y=None, LR=1e-4, L1=0, L2=0, epochs=1000, multi=False, ep=1e-9, gam=0.9, mu=0.9, lin=True, steps=None):

        if lin:
            for i in range(epochs):
                for a in range(0, x.shape[0], steps):
                    self.p = self.phi[a:(a+steps), :]@self.w

                    self.grad = (self.phi[a:(a+steps), :].T)@(self.p -
                                                              y[a:(a+steps), :])-L1*np.sign(self.w)-L2*self.w
                    self.G = gam*self.G+(1-gam)*self.grad**2
                    LRe = LR/np.sqrt(self.G+ep)

                    self.mom = mu*self.mom-LRe * \
                        ((self.phi[a:(a+steps), :].T@(self.p -
                                                      y[a:(a+steps), :]))-L1*np.sign(self.w)-L2*self.w)

                    self.w = self.w+self.mom

            if multi:
                self.errs = np.trace(
                    (self.p-y[a:(a+steps), :]).T@(self.p-y[a:(a+steps), :]))
                return("Error-", self.errs)
            else:
                self.errs = (self.p-y[a:(a+steps), :]
                             ).T@(self.p-y[a:(a+steps), :])
                return("Error-", self.errs)

        else:
            for i in range(epochs):
                for a in range(0, x.shape[0], steps):
                    self.p = self.activation(self.phi[a:(a+steps), :]@self.w)

                    self.grad = (self.phi[a:(a+steps), :].T)@(self.p -
                                                              y[a:(a+steps), :])-L1*np.sign(self.w)-L2*self.w
                    self.G = gam*self.G+(1-gam)*self.grad**2
                    LRe = LR/np.sqrt(self.G+ep)

                    self.mom = mu*self.mom-LRe * \
                        ((self.phi[a:(a+steps), :].T@(self.p -
                                                      y[a:(a+steps), :]))-L1*np.sign(self.w)-L2*self.w)

                    self.w = self.w+self.mom
                    self.pred = np.rint(self.p)
                    self.acc = np.mean(self.pred == y[a:(a+steps), :])

                if self.acc == 1.0:
                    print(i)
                    break
            if multi:
                self.errs = -np.sum(y[a:(a+steps), :]*np.log(self.p))
            else:
                self.errs = - \
                    np.sum(y[a:(a+steps), :]*np.log(self.p) +
                           (1-y[a:(a+steps), :])*np.log(1-self.p))
            self.predictions = np.rint(self.p)
            print(np.mean(self.predictions == y[a:(a+steps), :]))
            return(print("cost-", self.errs))

    def trainadam(self, x=None, y=None, LR=1e-4, L1=0, L2=0, epochs=1000, multi=False, ep=1e-9, gam=0.9, mu=0.9, lin=True, steps=None):
        if not steps:
            steps = x.shape[0]
        if lin:
            for i in range(epochs):
                for a in range(0, x.shape[0], steps):
                    self.p = (self.phi[a:(a+steps), :]@self.w)
                    self.mw = mu*self.mw + \
                        (1-mu)*self.phi[a:(a+steps),
                                        :].T@(self.p-y[a:(a+steps), :])
                    self.vw = gam*self.vw + \
                        (1-gam)*(self.phi[a:(a+steps), :].T @
                                 (self.p-y[a:(a+steps), :]))**2
                    self.mhat = self.mw/(1+mu**i)
                    self.vhat = self.vw/(1+gam**i)

                    LRw = LR/np.sqrt(self.vhat+ep)

                    self.w = self.w-LRw*self.mhat

                    self.pred = np.rint(self.p)
                    self.acc = np.mean(self.pred == y[a:(a+steps), :])
            if multi:
                self.errs = np.trace(
                    (self.p-y[a:(a+steps), :]).T@(self.p-y[a:(a+steps), :]))
                return("Error-", self.errs)
            else:
                self.errs = (self.p-y[a:(a+steps), :]
                             ).T@(self.p-y[a:(a+steps), :])
                return("Error-", self.errs)

        else:
            for i in range(epochs):
                for a in range(0, x.shape[0], steps):
                    self.p = self.activation(self.phi[a:(a+steps), :]@self.w)
                    self.mw = mu*self.mw + \
                        (1-mu)*self.phi[a:(a+steps),
                                        :].T@(self.p-y[a:(a+steps), :])
                    self.vw = gam*self.vw + \
                        (1-gam)*(self.phi[a:(a+steps), :].T @
                                 (self.p-y[a:(a+steps), :]))**2
                    self.mhat = self.mw/(1+mu**i)
                    self.vhat = self.vw/(1+gam**i)

                    LRw = LR/np.sqrt(self.vhat+ep)

                    self.w = self.w-LRw*self.mhat

                    self.pred = np.rint(self.p)
                    self.acc = np.mean(self.pred == y[a:(a+steps), :])

                if self.acc == 1.0:
                    print(i)
                    break
            if multi:
                self.errs = -np.sum(y[a:(a+steps), :]*np.log(self.p))
            else:
                self.errs = - \
                    np.sum(y[a:(a+steps), :]*np.log(self.p) +
                           (1-y[a:(a+steps), :])*np.log(1-self.p))
            self.predictions = np.rint(self.p)
            print(np.mean(self.predictions == y[a:(a+steps), :]))
            return(print("cost-", self.errs))
