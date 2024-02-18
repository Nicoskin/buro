import numpy as np
import matplotlib.pyplot as plt
import mylib as ml
import scipy.io as sp

def TED(data):
    """
    Вычисляет TED (Time Error Detector) для заданного массива данных.

    Аргументы:
    data -- массив данных, для которого нужно вычислить TED

    Возвращает:
    ted_plot -- массив значений TED для каждого временного сдвига

    Примечания:
    - TED (Time Error Detector) используется для оценки временной ошибки в сигнале.
    - Данные должны быть представлены в виде комплексных чисел.
    - Размер массива данных должен быть кратным 10.
    """
    err = np.zeros(len(data)//10, dtype="complex_")  # Создание массива нулей для хранения ошибок TED
    ted_plot = np.zeros(20, dtype="complex_")  # Создание массива нулей для хранения значений TED
    #data = np.roll(data, 0)  # Сдвиг данных на 0 позиций (без изменений)
    nsp = 10  # Количество семплов в одном символе
    buffer = np.zeros(len(data)//nsp, dtype="complex_")  
    
    for n in range(0, 20):  
        for ns in range(0, len(data)-(2*nsp+9), nsp):  # Цикл по значениям сдвига внутри символа
            real = (data.real[n+ns] - data.real[n+nsp+ns]) * data.real[n+nsp//2+ns]  # Вычисление вещественной части ошибки TED
            imag = (data.imag[n+ns] - data.imag[n+nsp+ns]) * data.imag[n+nsp//2+ns]  # Вычисление мнимой части ошибки TED
            err[ns//nsp] = real + imag  # Запись ошибки TED в массив
            
            buffer[ns//nsp] = err[ns//nsp]  # Запись ошибки TED в буфер
            
        ted_plot[n] = np.mean(buffer)  # Вычисление среднего значения ошибки TED и запись в массив TED
        buffer.fill(0)  # Очистка буфера ## НЕ знаю зачем, тк всё равно перезаписывается, но пусть будет )

    return ted_plot

data = sp.loadmat('recdata1702_5.mat')

h = list(data.values())
data = np.asarray(h[3])
data = np.ravel(data[:20000])
ml.cool_scatter(data, show_plot=False)

h1 = np.ones(10)
data = np.convolve(h1,data,"full")

plt.figure(1, figsize=(10, 5))
plt.title("Gardner TED")
plt.xlabel("tau")
plt.ylabel("e(ns)")
# for i in range(10):
#     plt.axvline(x=i, linestyle ="--", color ='red')

err = TED(data)
plt.plot(err.real, 'o-')
plt.grid()
plt.xticks(np.arange(0, 20))

sync_data = data[2::10] # начиная со второго, берём каждый 10

ml.cool_scatter(sync_data)
