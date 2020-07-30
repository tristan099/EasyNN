# lib
import numpy as np
import matplotlib.pyplot as plt

class EasyNN:

    def __init__(self, layers, costfunction):
        self.layers = layers
        self.costfunction = costfunction

    def predict(self, x=None, y=None, taskclassification=False, costfunction=None):
        self.layers[0].forward(x)
        for j in range(1, len(self.layers)):
            self.layers[j].forward(self.layers[j-1].z)

        if taskclassification:
            self.pred = np.rint(self.layers[len(self.layers)-1].z)
            self.acc = np.mean(self.pred == y)
            print("Accuracy-", self.acc)
        else:
            self.costs = self.costfunction(
                self.layers[len(self.layers)-1].z, y)
            print("Error-", self.costs)

    def train(self, x=None, y=None, LR=1e-4, epochs=1000, mu=0.9, k=0.5, steps=None, taskclassification=False, l1=0, l2=0, dropout=False, p=0):
        self.cost = []
        if not steps:
            steps = x.shape[0]

        for i in range(epochs):

            self.acctot = []

            if dropout:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forwarddrop(x[o:(o+steps), :], p)
                    for j in range(1, len(self.layers)):
                        self.layers[j].forwarddrop(self.layers[j-1].z, p)

                    self.layers[len(self.layers)-1].backward(LR,
                                                             y[o:(o+steps), :], mu, i, k, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backward(
                            self.layers[d+1].der, LR, mu, i, k, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
                    self.acctot = np.append(self.acctot, self.acc)
            else:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forward(x[o:(o+steps), :])
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z)

                    self.layers[len(self.layers)-1].backward(LR,
                                                             y[o:(o+steps), :], mu, i, k, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backward(
                            self.layers[d+1].der, LR, mu, i, k, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
                    self.acctot = np.append(self.acctot, self.acc)

        if taskclassification:
            print("Accuracy-", self.acc)
        else:
            print("Error-", self.costs)

    def trainnest(self, x=None, y=None, LR=1e-4, epochs=1000, mu=0.9, steps=None, taskclassification=False, l1=0, l2=0, dropout=False, p=0):
        self.cost = []
        if not steps:
            steps = x.shape[0]
        for i in range(epochs):

            if dropout:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forwarddrop(x[o:(o+steps), :], p)
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z, p)

                    self.layers[len(self.layers)-1].backwardnest(LR,
                                                                 y[o:(o+steps), :], mu, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardnesta(
                            self.layers[d+1].der, LR, mu, l1, l2)
                        self.layers[0].forward(x[o:(o+steps), :], p)
                        for k in range(1, len(self.layers)):
                            self.layers[k].forward(self.layers[k-1].z, p)
                        self.layers[len(self.layers)-1].backwardnest(LR,
                                                                     y[o:(o+steps), :], mu, l1, l2)
                        self.layers[d].backwardnestb(
                            self.layers[d+1].der, LR, mu, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])

            else:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forward(x[o:(o+steps), :])
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z)

                    self.layers[len(self.layers)-1].backwardnest(LR,
                                                                 y[o:(o+steps), :], mu, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardnesta(
                            self.layers[d+1].der, LR, mu, l1, l2)
                        self.layers[0].forward(x[o:(o+steps), :])
                        for k in range(1, len(self.layers)):
                            self.layers[k].forward(self.layers[k-1].z)
                        self.layers[len(self.layers)-1].backwardnest(LR,
                                                                     y[o:(o+steps), :], mu, l1, l2)
                        self.layers[d].backwardnestb(
                            self.layers[d+1].der, LR, mu, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])

        if taskclassification:
            print("Accuracy-", self.acc)
        else:
            print("Error-", self.costs)
        plt.plot(self.cost)
        plt.show()

    def adatrain(self, x=None, y=None, LR=1e-4, epochs=1000, mu=0.9, ep=1e-9, steps=None, taskclassification=False, l1=0, l2=0, dropout=False, p=0):
        self.cost = []
        if not steps:
            steps = x.shape[0]
        for i in range(epochs):
            if dropout:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forwarddrop(x[o:(o+steps), :], p)
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z, p)

                    self.layers[len(self.layers)-1].backwardada(LR,
                                                                mu, ep, y[o:(o+steps), :], l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardada(
                            self.layers[d+1].der, LR, mu, ep, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
            else:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forward(x[o:(o+steps), :])
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z)

                    self.layers[len(self.layers)-1].backwardada(LR,
                                                                mu, ep, y[o:(o+steps), :], l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardada(
                            self.layers[d+1].der, LR, mu, ep, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
        if taskclassification:
            print("Accuracy-", self.acc)
        else:
            print("Error-", self.costs)
        plt.plot(self.cost)
        plt.show()

    def adanesttrain(self, x=None, y=None, LR=1e-4, epochs=1000, mu=0.9, ep=1e-9, steps=None, taskclassification=False, l1=0, l2=0):
        self.cost = []
        if not steps:
            steps = x.shape[0]
        for i in range(epochs):
            for o in range(0, x.shape[0], steps):
                self.layers[0].forward(x[o:(o+steps), :])
                for j in range(1, len(self.layers)):
                    self.layers[j].forward(self.layers[j-1].z)

                self.layers[len(self.layers)-1].backwardadanest(LR,
                                                                mu, ep, y[o:(o+steps), :], l1, l2)
                for d in range(len(self.layers)-2, -1, -1):
                    self.layers[d].backwardadanesta(
                        self.layers[d+1].der, LR, mu, ep, l1, l2)
                    self.layers[0].forward(x[o:(o+steps), :])
                    for k in range(1, len(self.layers)):
                        self.layers[k].forward(self.layers[k-1].z)
                    self.layers[len(self.layers)-1].backwardadanest(LR,
                                                                    mu, ep, y[o:(o+steps), :], l1, l2)
                    self.layers[d].backwardadanestb(
                        self.layers[d+1].der, LR, mu, ep, l1, l2)

                self.costs = self.costfunction(
                    self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                self.cost = np.append(self.cost, self.costs)
                self.pred = np.rint(self.layers[len(self.layers)-1].z)
                self.acc = np.mean(self.pred == y[o:(o+steps), :])
        if taskclassification:
            print("Accuracy-", self.acc)
        else:
            print("Error-", self.costs)
        plt.plot(self.cost)
        plt.show()

    def rmstrain(self, x=None, y=None, LR=1e-4, epochs=1000, mu=0.9, ep=1e-9, gam=0.9, steps=None, taskclassification=False, l1=0, l2=0, dropout=False, p=0):
        self.cost = []
        if not steps:
            steps = x.shape[0]
        for i in range(epochs):

            if dropout:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forwarddrop(x[o:(o+steps), :], p)
                    for j in range(1, len(self.layers)):
                        self.layers[j].forwarddrop(self.layers[j-1].z, p)

                    self.layers[len(self.layers)-1].backwardrms(LR,
                                                                mu, ep, y[o:(o+steps), :], gam, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardrms(
                            self.layers[d+1].der, LR, mu, ep, gam, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
            else:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forward(x[o:(o+steps), :])
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z)

                    self.layers[len(self.layers)-1].backwardrms(LR,
                                                                mu, ep, y[o:(o+steps), :], gam, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardrms(
                            self.layers[d+1].der, LR, mu, ep, gam, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
        if taskclassification == True:
            print("Accuracy-", self.acc)
        else:
            print("Error-", self.costs)
        plt.plot(self.cost)
        plt.show()

    def rmsnesttrain(self, x=None, y=None, LR=1e-4, epochs=1000, mu=0.9, ep=1e-9, gam=0.9, steps=None, taskclassification=False, l1=0, l2=0, dropout=False, p=0):
        self.cost = []
        if not steps:
            steps = x.shape[0]
        for i in range(epochs):
            if dropout:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forwarddrop(x[o:(o+steps), :], p)
                    for j in range(1, len(self.layers)):
                        self.layers[j].forwarddrop(self.layers[j-1].z, p)

                    self.layers[len(self.layers)-1].backwardrmsnest(LR,
                                                                    mu, ep, y[o:(o+steps), :], gam, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardrmsnesta(
                            self.layers[d+1].der, LR, mu, ep, gam, l1, l2)
                        self.layers[0].forwarddrop(x[o:(o+steps), :], p)
                        for k in range(1, len(self.layers)):
                            self.layers[k].forwarddrop(self.layers[k-1].z, p)
                        self.layers[len(self.layers)-1].backwardrmsnest(LR,
                                                                        mu, ep, y[o:(o+steps), :], gam, l1, l2)
                        self.layers[d].backwardrmsnestb(
                            self.layers[d+1].der, LR, mu, ep, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
            else:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forward(x[o:(o+steps), :])
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z)

                    self.layers[len(self.layers)-1].backwardrmsnest(LR,
                                                                    mu, ep, y[o:(o+steps), :], gam, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardrmsnesta(
                            self.layers[d+1].der, LR, mu, ep, gam, l1, l2)
                        self.layers[0].forward(x[o:(o+steps), :])
                        for k in range(1, len(self.layers)):
                            self.layers[k].forward(self.layers[k-1].z)
                        self.layers[len(self.layers)-1].backwardrmsnest(LR,
                                                                        mu, ep, y[o:(o+steps), :], gam, l1, l2)
                        self.layers[d].backwardrmsnestb(
                            self.layers[d+1].der, LR, mu, ep, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])

        if taskclassification:
            print("Accuracy-", self.acc)
        else:
            print("Error-", self.costs)
        plt.plot(self.cost)
        plt.show()

    def adamtrain(self, x=None, y=None, LR=1e-4, epochs=1000, mu=0.9, ep=1e-9, gam=0.9, taskclassification=False, steps=None, l1=0, l2=0, dropout=False, p=0):
        self.cost = []
        if not steps:
            steps = x.shape[0]
        for i in range(epochs):

            if dropout:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forwarddrop(x[o:(o+steps), :], p)
                    for j in range(1, len(self.layers)):
                        self.layers[j].forwarddrop(self.layers[j-1].z, p)

                    self.layers[len(self.layers)-1].backwardadam(LR,
                                                                 mu, ep, gam, y[o:(o+steps), :], i, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardadam(
                            self.layers[d+1].der, LR, mu, ep, gam, i, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
            else:
                for o in range(0, x.shape[0], steps):
                    self.layers[0].forward(x[o:(o+steps), :])
                    for j in range(1, len(self.layers)):
                        self.layers[j].forward(self.layers[j-1].z)

                    self.layers[len(self.layers)-1].backwardadam(LR,
                                                                 mu, ep, gam, y[o:(o+steps), :], i, l1, l2)
                    for d in range(len(self.layers)-2, -1, -1):
                        self.layers[d].backwardadam(
                            self.layers[d+1].der, LR, mu, ep, gam, i, l1, l2)

                    self.costs = self.costfunction(
                        self.layers[len(self.layers)-1].z, y[o:(o+steps), :])
                    self.cost = np.append(self.cost, self.costs)
                    self.pred = np.rint(self.layers[len(self.layers)-1].z)
                    self.acc = np.mean(self.pred == y[o:(o+steps), :])
        if taskclassification:
            print("Accuracy-", self.acc)
        else:
            print("Error-", self.costs)
        plt.plot(self.cost)
        plt.show()
