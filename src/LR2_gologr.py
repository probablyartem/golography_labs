import numpy as np 
import cv2
import pylab
import matplotlib.pyplot as plt


def to_2d_center_f(input_image, centered = True):
    (image_height, image_width) = np.shape(input_image)
    centered_image = np.zeros(shape=(image_height, image_width))
    for i in range(image_height):
        empty_array = []
        for j in range(image_width):
            if centered:
                empty_array.append(input_image[i][j]*((-1)**(i+j)))
            else:
                empty_array.append(input_image[i][j].real*((-1)**(i+j)))
        centered_image[i] = empty_array
    return np.array(centered_image)

def filter_lf_2d(transformed_image, shift):
    transformed_image = transformed_image.copy()
    array_height, array_width = np.shape(transformed_image)

    # transformed_image[:array_height // 2 - shift[0], :] *= 0 
    # transformed_image[array_height // 2 + shift[0] + 1:, :] *= 0 
    transformed_image[:, :array_height // 2 -shift[1]] *= 0 
    transformed_image[:, array_width // 2 + shift[1] + 1:] *= 0 
    return np.array(transformed_image)

def filter_hor(transformed_image, strip_height):
    transformed_image = transformed_image.copy()
    array_height, array_width = np.shape(transformed_image)

    # Оставить только полоску по горизонтали
    transformed_image[strip_height+1:array_height - strip_height, :] = 0

    return np.array(transformed_image)


def filter_diagonal(transformed_image, strip_width):
    transformed_image = transformed_image.copy()
    array_height, array_width = np.shape(transformed_image)
    
    # Оставить только диагональную полосу
    for i in range(array_height):
        for j in range(array_width):
           if i - j < -strip_width or i - j > strip_width:
                    transformed_image[i][j] = 0

    return np.array(transformed_image)


image = cv2.imread('reshetka5.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fourier_image = np.fft.fft2(gray_image)

plt.imshow(gray_image, cmap='gray')
plt.show()


fourier_image_shifted = np.fft.fftshift(fourier_image)


plt.imshow(np.log(np.abs(fourier_image_shifted) + 1), cmap='gray')
plt.show()

strip_width = 1  # Ширина полоски по горизонтали
shift = [strip_width, strip_width]

hor_fourier_image = filter_lf_2d(fourier_image_shifted, shift)

# Отобразите отфильтрованный Фурье-образ
plt.imshow(np.log(np.abs(hor_fourier_image) + 1), cmap='gray')
plt.show()


reconstructed_image = np.fft.ifft2(np.fft.ifftshift(hor_fourier_image)).real


plt.imshow(reconstructed_image, cmap='gray')
plt.show()

strip_width = 1  # Ширина диагональной полосы
diagonal_fourier_image = filter_diagonal(fourier_image_shifted, strip_width)

plt.imshow(np.log(np.abs(diagonal_fourier_image) + 1), cmap='gray')
plt.show()


reconstructed_image = np.fft.ifft2(np.fft.ifftshift(diagonal_fourier_image)).real


plt.imshow(reconstructed_image, cmap='gray')
plt.show()
