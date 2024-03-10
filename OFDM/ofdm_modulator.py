import matplotlib.pyplot as plt
import numpy as np




def QPSK(bit_mass):
	ampl = 2**14
	if (len(bit_mass) % 2 != 0):
		print("QPSK:\nError, check bit_mass length", len(bit_mass))
		raise "error"
	else:
		sample = [] # массив комплексных чисел
		for i in range(0, len(bit_mass), 2):
			b2i = bit_mass[i]
			b2i1 = bit_mass[i+1]
			real = (1 - 2 * b2i) / np.sqrt(2)
			imag = (1 - 2 * b2i1) / np.sqrt(2)
			sample.append(complex(real, imag))
		sample = np.asarray(sample)
		sample = sample * ampl
		return sample

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return np.asarray(list(map(int,bits.zfill(8 * ((len(bits) + 7) // 8)))))


def delete_CP(rx_ofdm, num_carrier, cp): # удаляет циклический префикс и делает преобразование фурье по кол-ву поднесущих

    rx_sig_de = np.zeros(0)

    for i in range(len(rx_ofdm) // num_carrier):
        del_cp = rx_ofdm[i * (cp + num_carrier)+cp:(i + 1) * (cp+num_carrier)]
        de_symbol = np.fft.fft(del_cp, num_carrier)
        rx_sig_de = np.concatenate([rx_sig_de, de_symbol])

    return rx_sig_de


def gen_ofdm_symbols(qpsk1,num_carrier,cp,zero): #  создаем офдм сиволы и добавляем циклический префикс

    ofdm_symbols = np.zeros(0, dtype=np.complex128)
    zero_guard = np.zeros(zero)
    for i in range(len(qpsk1)//num_carrier):
        ofdm_symbol = np.fft.ifft(qpsk1[i * num_carrier : (i + 1) * num_carrier], num_carrier)
        ofdm_symbols = np.concatenate([ofdm_symbols, zero_guard, ofdm_symbol[-cp:], ofdm_symbol, zero_guard])
        
    return ofdm_symbols


def correlat_ofdm(rx_ofdm, cp, num_carrier): # находит начало OFDM символа, по средству корреляции по циклическому префиксу 
    max = 0
   
    for j in range(len(rx_ofdm)):

        corr_sum = np.correlate(rx_ofdm[:cp], rx_ofdm[(num_carrier - cp):num_carrier])
        if corr_sum > max:
            max = corr_sum
            index = j
        rx_ofdm= np.roll(rx_ofdm,-1)

    return index # возвращает индекс начала сообщения

def add_pilot(qpsk,pilot,step_pilot): # добавление pilot с заданым шагом (pilot - переменная, одно значение пилота)

    step_pilot -= 1
    newarr = []
    newarr.append(pilot)

    for i in range( len(qpsk) ):
        newarr.append( qpsk[i] )
        
        if (i + 1) % step_pilot == 0:
            newarr.append(pilot)
    
    return np.asarray(newarr)

def del_pilot(ofdm,pilot_carrier):

    ofdm = np.delete(ofdm,pilot_carrier)

    return ofdm


def PLL(conv):
    mu = 1# коэфф фильтра 
    theta = 0.40 # начальная фаза
    phase_error = np.zeros(len(conv))  # фазовая ошибка
    output_signal = np.zeros(len(conv), dtype=np.complex128)

    for n in range(len(conv)):
        theta_hat = np.angle(conv[n])  # оценка фазы
        #print(theta_hat)
        phase_error[n] = theta_hat - theta  # фазовая ошибка
        output_signal[n] = conv[n] * np.exp(-1j * theta)  # выходной сигнал
        theta = theta + mu * phase_error[n]  # обновление

    return output_signal

def draw_ofdm_symbols(ofdm_symbols, num_carrier, cp, zero_guard):
    draw_symbols = np.arange(0,len(ofdm_symbols),cp + num_carrier + (2*zero_guard))

    plt.figure(1)    # русуем границы офдм символа с циклическим префиксом
    plt.title("Циклический префикс + OFDM-символы + защитные нули")
    plt.stem(abs(ofdm_symbols))

    for x in draw_symbols:
        plt.axvline(x , color = 'r')     

def OFDM_MODULATOR(bit, num_carrier, cp, zero_guard):
    pilot = complex(1,1)

    qpsk1 = QPSK(bit)

    #pilot_carrier = np.arange(0,len(qpsk1),step_pilot)# for del pilot

    print("len qpsk = ",len(qpsk1))


    added_pilot = add_pilot(qpsk1,pilot,step_pilot)

    ofdm_symbols = gen_ofdm_symbols(added_pilot, num_carrier,cp, zero_guard)


    print("len ofdm_simbols = ",len(ofdm_symbols))


    return ofdm_symbols

        


bit = text_to_bits("I am sure learning foreign languages is very important nowadays. People start learning a foreign language, because they want to have a better job")
print(len(bit))


num_carrier = 64
cp = 16
step_pilot = 10
zero_guard = 5


ofdm_symbols = OFDM_MODULATOR(bit,num_carrier,cp,zero_guard)

draw_ofdm_symbols(ofdm_symbols,num_carrier,cp,zero_guard)

plt.show() 