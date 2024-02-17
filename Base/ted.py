import numpy as np
import matplotlib.pyplot as plt
import mylib as ml
import scipy. io as sp

def TED(data):
    err = np.zeros(len(data)//10, dtype = "complex_")
    ted_plot = np.zeros(20, dtype = "complex_")
    data = np.roll(data,0)
    nsp = 10
    buffer = np.zeros(len(data)//nsp, dtype = "complex_")
    
    for n in range(0,20):
        for ns in range(0,len(data)-(2*nsp+9),nsp):
            real = (data.real[n+ns] - data.real[n+nsp+ns]) * data.real[n+nsp//2+ns]
            imag = (data.imag[n+ns] - data.imag[n+nsp+ns]) * data.imag[n+nsp//2+ns]
            err[ns//nsp] = real + imag 
            
            buffer[ns//nsp] = err[ns//nsp]
            
        ted_plot[n] = np.mean(buffer)
        buffer.fill(0)

    return ted_plot

def Get_Index(qpsk, index): # вычленяет семпл выбранный глазковой диаграммой 
    decode_symbol = []
    for i in range(index,len(qpsk),10):
        #print()
        decode_symbol.append(qpsk[i])
    return np.asarray(decode_symbol)

data = sp.loadmat('C:\\Users\\Ivan\\Desktop\\lerning\\buro1440\\buro\\Base\\recdata1702_5.mat')

h = list(data.values())
data = np.asarray(h[3])
data = np.ravel(data[:20000])
ml.cool_scatter(data)


h1 = np.ones(10)
data = np.convolve(h1,data,"full")



plt.figure(1)
plt.title("Gardner TED")
plt.xlabel("tau")
plt.ylabel("e(ns)")
for i in range(10):
    plt.axvline(x=i, linestyle ="--", color ='red')

err = TED(data)
plt.plot(err.real)

sync_data = Get_Index(data,2)

ml.cool_scatter(sync_data)



plt.show()