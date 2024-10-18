import cv2
import numpy as np

# Função para contar as moedas em uma imagem
def contar_moedas(image_path):
    # Carregar a imagem em tons de cinza
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar um filtro Gaussiano para suavizar a imagem e reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Binarizar a imagem usando a limiarização adaptativa
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverter a imagem para que as moedas estejam em branco (foreground)
    binary = cv2.bitwise_not(binary)

    # Aplicar operações morfológicas para melhorar a detecção das moedas
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Aplicar distance transform para identificar os centros das moedas
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Identificar a região de fundo
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Rotular os componentes conectados
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Aumentar os rótulos para que o fundo seja 1
    markers[unknown == 255] = 0

    # Aplicar o algoritmo watershed para separar as moedas
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # Contornos em vermelho

    # Contar o número de moedas detectadas (excluindo o fundo)
    num_moedas = len(np.unique(markers)) - 2  # Desconta fundo e bordas (-1)

    # Mostrar resultados
    cv2.imshow('Imagem Original', image)
    cv2.imshow('Imagem Binarizada', binary)
    cv2.imshow('Distance Transform', dist_transform)
    cv2.imshow('Detecção de Moedas', sure_fg)
    cv2.imshow('Resultado Final', image)

    return num_moedas

# Dicionário de imagens e número verdadeiro de moedas
image_set = {
    'images/desafio_1/coins6.jpg': 6,
    'images/desafio_1/coins10.jpg': 10,
    'images/desafio_1/coins30.jpg': 30,
    'images/desafio_1/coins50.jpg': 50,
    'images/desafio_1/coins100.jpg': 100,
    'images/desafio_1/coins101.jpg': 101
}

# Avaliação das estimativas
total_erro_relativo = 0
total_quadratic_error = 0

for image_name, true_count in image_set.items():
    estimated_count = contar_moedas(image_name)
    erro_relativo = abs(true_count - estimated_count) / true_count * 100
    quadratic_error = (true_count - estimated_count) ** 2

    print(f"Imagem: {image_name}")
    print(f"Número verdadeiro de moedas: {true_count}")
    print(f"Número estimado de moedas: {estimated_count}")
    print(f"Erro relativo: {erro_relativo:.2f}%")
    print()

    total_erro_relativo += erro_relativo
    total_quadratic_error += quadratic_error

# Cálculo dos resultados globais
media_erro_relativo = total_erro_relativo / len(image_set)
media_quadratic_error = total_quadratic_error / len(image_set)

print(f"Média dos erros relativos: {media_erro_relativo:.2f}%")
print(f"Erro quadrático médio: {media_quadratic_error:.2f}")
