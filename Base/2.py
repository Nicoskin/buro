import numpy as np
import matplotlib.pyplot as plt
import mylib as ml





def test_plot(x):
    plt.plot(x[:300].real, 'o-')
    plt.plot(x[:300].imag, 'o-')
    plt.show()

bit = ml.gen_rand_bits(200)
sigQpsk = ml.qpsk(bit, 1) 
sig = np.repeat(sigQpsk, 10) # 10 сэмплов на символ
noise = np.random.normal(0, 0.05, len(sig)) + 1j * np.random.normal(0, 0.05, len(sig))
sigTx = sig + noise

sigRx = np.convolve(sigTx, np.ones(10)) / 10 # свёртка + нормирование 

ml.cool_scatter(sigTx)  ### График QPSK c шумом
ml.cool_scatter(sigRx)  ### График QPSK после свёртки (квадрат с крестом)

test_plot(sigTx) ### График QPSK c шумом
test_plot(sigRx) ### График QPSK после свёртки (квадрат с крестом)


ml.eye_pattern(sigRx)   ### Глазковая диаграмма