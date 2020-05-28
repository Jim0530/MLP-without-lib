import numpy as np
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        self.dx= None
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        self.dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return self.dx
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid的輸出
        self.t = None  # 訓練資料
        self.dx= None
    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = np.sum(0.5*(self.y-self.t)**2,axis=1)
            
        return self.loss

    def backward(self, dout=1):
        self.dx = (self.y - self.t) * dout 
        return self.dx
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        self.dx  = None
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return self.out

    def backward(self, dout):
        self.dx = dout * (1.0 - self.out) * self.out
        return self.dx
class Softmax():
    def __init__(self):
        self.params,self.grads=[],[]
    def forward(self,x):
        e=np.exp(x-np.max(x))
        return e/(np.sum(e))
class loaddata():
    def __init__(self,file):
        data = np.loadtxt((file+'.txt'))
        dim  = data.shape[1]-1
        self.x,self.t=data[:,0:dim],data[:,dim:]
        self.label_type=np.unique(self.t)
    def onehotencode(self):
        t_onehot=np.zeros(([self.t.shape[0],len(self.label_type)]))
        for place,value in enumerate(self.label_type):
            letter=[0 for i in range(len(self.label_type))]
            letter[place]=1
            for index,num in enumerate(self.t):
                    if num==value:
                        t_onehot[index]=letter
        return self.x,t_onehot
    def normal(self):
        return self.x,self.t
class SoftmaxCrossEntropy():
    def __init__(self,t):
        self.t=t
        self.e=None
        self.N=None
        self.loss=None
        self.out=None
    def forward(self,x):
        self.N=x.shape[0]
        self.e=np.exp(x-np.max(x))
        self.out=self.e/(np.sum(self.e,axis=1,keepdims=True))
        eout=np.clip(self.out,1e-12,1.-(1e-12))
        self.loss = -np.sum(self.t*np.log(eout+1e-9))/self.N
        return self.loss
    def backward(self,dout=1):
        grad=self.out.copy()
        t_onedim=np.argmax(self.t,axis=1)
        grad[range(self.t.shape[0]),t_onedim] -=1
        grad=grad/(self.t.shape[0])
        return grad
class Relu():
    def __init__(self):
        None
    def forward(self,x):
        return np.maximum(0,x) # 當然 np.maximum 接受的兩個參數，也可以大小一致
        # 或者更為準確地說，第二個參數只是一個單獨的值時，其實是用到了維度的 broadcast 機制；
    def backward(self,dout):
        grad=dout>0
        return grad*dout