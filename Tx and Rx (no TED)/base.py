import numpy as np
import matplotlib.pyplot as plt
import mylib as ml
import mylib.test as mlt

###
### Создание данных
###
bits = ml.str_to_bits('A small text to send to the radio channel') # 328 бита
sigQpsk = ml.qpsk(bits) 
barkerCode = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]
barkerBpsk = ml.bpsk(barkerCode)
# barkerBpsk = barkerBpsk * np.exp(1j * np.pi/4) # Доворот на синхры 45 градусов
signal = np.concatenate((barkerBpsk, sigQpsk)) # 177 символа (164 без баркера)
signalRepeat = np.repeat(signal, 10) # 1770 сэмпла

###
### Работа с SDR
###
sdr = ml.sdr_settings("ip:192.168.3.1", 2300e6+(2e6*2), 1000, 1e6, 0, 0) # type: ignore
ml.tx_sig(sdr,signalRepeat) 
rx = ml.rx_cycles_buffer(sdr, 4) # берёт 4000 отсчётов
rxMax = max(rx.real)
rx = rx / rxMax # Нормировка

symbolLength = 10 
rxConvolve = np.convolve(rx, np.ones(symbolLength)) / 10 # Свёртка


### тут TED, Loop Filter ...
### и мы определяем какой отсчёт брать
maxe = np.argmax(rxConvolve) % 10 # типа TED
rxSymbols = rxConvolve[maxe::10] # типа TED


rxClear = ml.bpsk_synchro(rxSymbols, barkerCode, debug=True) # разворот на правильный угол по синхре
rxClear = rxClear[13:] # отрезаем баркер

# Демодуляция и декодирование
message = ml.dem_qpsk(rxClear)
text = ml.bits_to_str(message)
print('\nRx -->',bits)
print('Rx message --> A small text to send to the radio channel\n')
print('Tx -->',message)
print('Tx message -->',text)

ml.cool_plot(rx, title="Rx signal")
ml.cool_plot(rxConvolve, title="После свёртки")
ml.cool_plot(rxClear, title="Повёрнутый")

ml.cool_scatter(rx, title="Rx signal")
ml.cool_scatter(rxSymbols, title="Символы")
ml.cool_scatter(rxClear, title="Повёрнутый")


plt.show()