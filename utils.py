import os
import numpy
import numpy as np
from scipy.stats import norm
numpy.seterr(all='ignore')

def get_day_set(day, num_files = np.inf, root=''):
    path = f"{root}/{day}/"
    files = os.listdir(path)
    files.sort()
    ds_min = np.inf
    ds_max = -np.inf
    i = 0
    labels = None
    ds = []
    for f in files:
        if "part" in f:
            if i == num_files:
                break
            i += 1
            m = np.load(path+f)
            ds.append(m)
    ds = np.concatenate(ds)
    ds_min = np.min(ds,axis=0)
    ds_max = np.max(ds,axis=0)
    if day != "monday":
        labels = np.load(path+"labels.npy")[:ds.shape[0]]

    return ds, labels, ds_min, ds_max

def get_ds(root_dir, num_files=np.inf):
    import numpy as np
    ds = []
    y = []
    monday_ds_size = 0
    for day in [ "monday", "tuesday", "wednesday", "thursday", "friday" ]:
        x, labels, _, _ = get_day_set(day, num_files=num_files, root=root_dir)
        if day != "monday":
            y.append(labels)
        else:
            monday_ds_size = x.shape[0]
        ds.append(x)

    X = np.concatenate(ds, axis=0)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_normalized = (X - X_min)/(X_max - X_min+0.000001)
    y = np.concatenate(y, axis=0)
    return X_normalized, y, monday_ds_size

def pdf(x,mu,sigma): #normal distribution pdf
    x = (x-mu)/sigma
    return numpy.exp(-x**2/2)/(numpy.sqrt(2*numpy.pi)*sigma)

def invLogCDF(x,mu,sigma): #normal distribution cdf
    x = (x - mu) / sigma
    return norm.logcdf(-x) #note: we mutiple by -1 after normalization to better get the 1-cdf

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return numpy.tanh(x)

def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

class rollmean:
    def __init__(self,k):
        self.winsize = k
        self.window = numpy.zeros(self.winsize)
        self.pointer = 0

    def apply(self,newval):
        self.window[self.pointer]=newval
        self.pointer = (self.pointer+1) % self.winsize
        return numpy.mean(self.window)

# probability density for the Gaussian dist
# def gaussian(x, mean=0.0, scale=1.0):
#     s = 2 * numpy.power(scale, 2)
#     e = numpy.exp( - numpy.power((x - mean), 2) / s )

#     return e / numpy.square(numpy.pi * s)

