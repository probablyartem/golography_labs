# import numpy as np 
# import matplotlib.pyplot as plt
# z = np.linspace(-0.002,0.002,400)
# f = 1.1 
# lyambda = 0.632*10**(-6)
# x = 2.21*10**(-3)
# sinc = np.sin(x*np.pi*z/(lyambda*f))/(x*np.pi*z/(lyambda*f))
# sinc2 = np.sinc(x)**2

# fff = np.fft.fftshift(np.fft.fft(sinc))
# # fff2 = np.fft.fftshift(np.fft.fft(sinc2))
# freq = np.fft.fftshift(np.fft.fftfreq(len(z)))

# plt.subplot(2, 1, 1)
# plt.plot(z, (sinc))
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(freq, np.abs(fff))
# plt.grid(True)
# plt.show()


import numpy as np 
import matplotlib.pyplot as plt
x = np.linspace(-10,10,10000)
f = 1.1 
lyambda = 0.632*10**(-6)
z = 314.57*10**(-6)
sinc = np.sin(x*np.pi*z/(lyambda*f))/(x*np.pi*z/(lyambda*f))
sinc2 = np.sinc(x)**2
    
fff = np.fft.fftshift(np.fft.fft(sinc2))
# fff2 = np.fft.fftshift(np.fft.fft(sinc2))
freq = np.fft.fftshift(np.fft.fftfreq(len(x)))

plt.subplot(2, 1, 1)
plt.plot(x, (sinc2))
plt.ylabel("Распределеление амплитуды, отн. ед")
plt.xlabel("x, мм")
plt.grid(True)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freq, np.abs(fff))
plt.ylabel("Фурье-образ, отн. ед")
plt.xlabel("d, мм")
plt.grid(True)
plt.grid(True)
plt.show()