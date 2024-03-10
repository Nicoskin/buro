import numpy as np
import matplotlib.pyplot as plt
import mylib as ml
import mylib.test as mlt

# TODO: протестировать при передаче с одной сдр на другую
rx = np.load("") # Прописать свой путь к файлу 
rxMax = max(rx.real)
rx = rx / rxMax # Нормировка
symbolLength = 10 
rxConvolve = np.convolve(rx, np.ones(symbolLength)) / 10 # Свёртка

def FLL(conv):
    mu = 0.01
    omega = 0.32  # TODO: нужно протестировать для разных сигналов, пока непонятно, работает ли этот коэффициент для всех QPSK-сигналов
    freq_error = np.zeros(len(conv))
    output_signal = np.zeros(len(conv), dtype=np.complex128)

    for n in range(len(conv)):
        angle_diff = np.angle(conv[n]) - np.angle(output_signal[n-1]) if n > 0 else 0
        freq_error[n] = angle_diff / (2 * np.pi)
        omega = omega + mu * freq_error[n]
        output_signal[n] = conv[n] * np.exp(-1j * omega)
    return output_signal


def PLL(conv):
    mu = 1
    theta = 1
    phase_error = np.zeros(len(conv))  
    output_signal = np.zeros(len(conv), dtype=np.complex128)

    for n in range(len(conv)):
        theta_hat = np.angle(conv[n])
        phase_error[n] = theta_hat - theta  
        output_signal[n] = conv[n] * np.exp(-1j * theta)  
        theta = theta + mu * phase_error[n]
    return output_signal

def TED_loop_filter(data):
    BnTs = 0.01 
    Nsps = 10
    C = np.sqrt(2)
    Kp = 1
    teta = ((BnTs)/(Nsps))/(C + 1/(4*C))
    K1 = (-4*C*teta)/((1+2*C*teta+teta**2)*Kp)
    K2 = (-4*teta**2)/((1+2*C*teta+teta**2)*Kp)
    print("K1 = ", K1)
    print("K2 = ", K2)
    err = np.zeros(len(data)//10, dtype = "complex_")
    data = np.roll(data,-0)
    nsp = 10
    p1 = 0
    p2 = 0
    n = 0
    mass_cool_index = []
    mass_id = []
    for ns in range(0,len(data)-(2*nsp),nsp):
        real = (data.real[nsp+ns+n] - data.real[ns+n]) * data.real[n + (nsp)//2+ns]
        imag = (data.imag[nsp+ns+n] - data.imag[ns+n] ) * data.imag[n + (nsp)//2+ns]
        err[ns//nsp] = np.mean(real + imag)
        error = err.real[ns//nsp]
        p1 = error * K1
        p2 = p2 + p1 + error * K2
        while(p2 > 1):
            p2 = p2 - 1
        
        n = round(p2*10)  
        n1 = n+ns+nsp   
        mass_cool_index.append(n1)
        mass_id.append(n)

    mass1 = np.asarray(mass_cool_index)
    mass = np.asarray(mass_id)
    return mass1


Ted_index = TED_loop_filter(rxConvolve)
### тут TED, Loop Filter ...
### и мы определяем какой отсчёт брать
rxАfterTED = rxConvolve[Ted_index]
# Работа ФАПЧ
rxАfterTED = FLL(PLL(rxАfterTED))

ml.cool_plot(rxАfterTED, title="rxАfterTED")
ml.cool_scatter(rx, title="Принятый сигнал")
ml.cool_scatter(rxАfterTED, title="Сигнал после символьной синхронизации и ФАПЧ")

plt.show()