from MLPmodel import Model,accuracy,draw
from Layer import loaddata
from sklearn.model_selection import train_test_split
from tkinter import messagebox,Entry,Tk,StringVar,Button,Listbox
import numpy as np

def Sucess():
    messagebox.showinfo("Sucess", "Sucess")
def LearningRate():
    global lr
    lr = E1.get()
    Sucess()
def Convergence():
    global epoch
    epoch = E2.get()
    Sucess()
def windowdestroy():
    window.destroy()
def selection():
    global data_name
    data_name = lb.get(lb.curselection())
    Sucess()
def selection2():
    global algor
    algor = lb2.get(lb2.curselection())
    Sucess()
def train():
    data=loaddata(data_name)
    x,t=data.onehotencode()
    I,H,H1,O=x.shape[1],12,8,t.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(x,t,test_size=1/3,random_state=6)
    W1=0.01*np.random.randn(I,H)
    W2=0.01*np.random.randn(H,H1)
    W3=0.01*np.random.randn(H1,O)
    B1=np.zeros(H)
    B2=np.zeros(H1)
    B3=np.zeros(O)
    model=Model(W1,W2,W3,B1,B2,B3,y_train,algor)
    best_params=[]
    for i in model.params_layer:
        best_params.append(np.zeros_like(i))
    loss=0
    min_loss=100000
    for i in range(int(epoch)):
        loss=model.forward(x_train)
        if loss<min_loss:
            min_loss=loss
            for i in range(len(best_params)):
                best_params[i]=model.params_layer[i].copy()
        model.backward()
        model.update()
    model.layers[0].params[0]=best_params[0]
    model.layers[0].params[1]=best_params[1]
    model.layers[2].params[0]=best_params[2]
    model.layers[2].params[1]=best_params[3]
    model.layers[4].params[0]=best_params[4]
    model.layers[4].params[1]=best_params[5]   
    train_answer=model.predict(x_train)
    y_train=np.argmax(y_train,axis=1)
    ac_train=accuracy(y_train,train_answer)
    print('train data predictions\n',train_answer)
    print('train data correct answer\n',y_train)
    print('train data accracy=',ac_train)
    test_answer=model.predict(x_test)
    y_test=np.argmax(y_test,axis=1)
    ac_test=accuracy(y_test,test_answer)
    print('test data predictions\n',test_answer)
    print('test data correct answer\n',y_test)
    print('test data accracy=',ac_test)
    draw(x_train,y_train,train_answer,x_test,y_test,test_answer)
window = Tk()
window.geometry('1000x1000')
window.title('Nueral Network Settings')
var2 = StringVar()
var2.set(('perceptron1','perceptron2','perceptron3','perceptron4','2Ccircle1','2Circle2','2CloseS','2CloseS2','2CloseS3','2cring','2CS','2Hcircle1','2ring','IRIS','5CloseS1','8OX',
         'C3D','C10D','4satellite-6','winie','xor','Number'))
var1 = StringVar()
var1.set(('RMSprop','Adam','Momentum'))
E1 = Entry(window)
E1.pack()
L= Button(window, text = "Input LearningRate", command = LearningRate)
L.pack()
E2 = Entry(window)
E2.pack()
C= Button(window, text = "Input how many epochs", command = Convergence)
C.pack()
lb = Listbox(window, listvariable=var2)
lb.pack()
N=Button(window, text = "Press the Button to Save what u select from the list", command = selection)
N.pack()
lb2 = Listbox(window, listvariable=var1)
lb2.pack()
Z=Button(window, text = "Press the Button to Save what u select from the list", command = selection2)
Z.pack()
train=Button(window,text='start training',command=train)
train.pack()
End=Button(window, text = "Finish", command = windowdestroy)
End.pack()
window.mainloop()
