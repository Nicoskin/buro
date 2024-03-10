import numpy as np
import matplotlib.pyplot as plt
import mylib as ml
import mylib.test as mlt


def TED_loop_filter(data): #ted loop filter 
    BnTs = 0.01 
    Nsps = 10
    C = np.sqrt(2)/2
    Kp = 1
    teta = ((BnTs)/(Nsps))/(C + 1/(4*C))
    K1 = (-4*C*teta)/((1+2*C*teta+teta**2)*Kp)
    K2 = (-4*teta**2)/((1+2*C*teta+teta**2)*Kp)
    print("K1 = ", K1)
    print("K2 = ", K2)
    #K1_2 = (1/Kp)*((((4*C)/(Nsps**2))*((BnTs/(C + (1/4*C)))**2))/(1 + ((2 * C)/Nsps)*(BnTs/(C + (1/(4*C))))+(BnTs/(Nsps*(C+(1/4*C))))**2))
    err = np.zeros(len(data)//10, dtype = "complex_")
    data = np.roll(data,-0)
    nsp = 10
    p1 = 0
    p2 = 0
    n = 0
    mass_cool_inex = []
    mass_id = []
    for ns in range(0,len(data)-(2*nsp),nsp):
        #real = (data.real[ns+n] - data.real[nsp+ns+n]) * data.real[n+(nsp)//2+ns]
        #imag = (data.imag[ns+n] - data.imag[nsp+ns+n]) * data.imag[n+(nsp)//2+ns]
        real = (data.real[nsp+ns+n] - data.real[ns+n]) * data.real[n + (nsp)//2+ns]
        imag = (data.imag[nsp+ns+n] - data.imag[ns+n] ) * data.imag[n + (nsp)//2+ns]
        err[ns//nsp] = real + imag
        error = err.real[ns//nsp]
        p1 = error * K1
        p2 = p2 + p1 + error * K2
        #print(ns ," p2 = ",p2)  
        while(p2 > 1):
            #print(ns ," p2 = ",p2)
            p2 = p2 - 1
        #while(p2 < -1):
            #print(ns ," p2 = ",p2)
            #p2 = p2 + 1
        
        n = round(p2*10)  
        n1 = n+ns+nsp   
        mass_cool_inex.append(n1)
        mass_id.append(n)

    #mass_cool_inex = [math.ceil(mass_cool_inex[i]) for i in range(len(mass_cool_inex))]
    mass1 = np.asarray(mass_cool_inex)
    mass = np.asarray(mass_id)
    plt.subplot(2,1,1)
    plt.plot(err) 
    plt.subplot(2,1,2)
    plt.plot(mass)   
    
    return mass1


bits = ml.str_to_bits('A small text to send to the radio channel') # 328 бита
sigQpsk = ml.qpsk(bits) 
#barkerCode = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]
#barkerBpsk = ml.bpsk(barkerCode)
#signal = np.concatenate((barkerBpsk, sigQpsk)) # 177 символа (164 без баркера)
signalRepeat = np.repeat(sigQpsk, 10) # 1770 сэмпла

sdr = ml.sdr_settings("ip:192.168.2.1", 2000e6, 1000, 1e6, 0, 0) # type: ignore
ml.tx_sig(sdr,signalRepeat) 
rx = ml.rx_cycles_buffer(sdr, 4) # берёт 4000 отсчётов

rxMax = max(rx.real)
rx = rx / rxMax # Нормировка

#np.save('rx_50k_4', rx)

# r1 = np.load('rx_50k_0.npy')
ml.cool_scatter(rx)



h1 = np.ones(10)
data = np.convolve(h1,rx,"full")/10

plt.figure(2)
mass_ind = TED_loop_filter(data)
new_data = data[mass_ind]
print(len(new_data))

#plt.figure(3)
plt.title("QPSK sync")

ml.cool_scatter(new_data)


plt.show()