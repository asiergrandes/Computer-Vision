import cv2
import numpy as np
import matplotlib.pyplot as plt

#########################################################
# 1.  Concept of color and color spaces, RGB, HSV and Lab
#########################################################

# a)
'''
def plot_channels(image, color_space, image_name):

    # Separar los canales de la imagen
    channels = cv2.split(image)
    names = {"HSV": ["H", "S", "V"], "Lab": ["L", "a", "b"]}[color_space]

    # Mostrar cada canal en una figura
    plt.figure(figsize=(12, 4))
    for i, ch in enumerate(channels):
        plt.subplot(1, 3, i+1)
        plt.imshow(ch, cmap='gray')  # Escala de grises para un solo canal
        plt.title(f"{image_name} {color_space} - {names[i]}")
        plt.axis("off")
    plt.show()


img_yosemite = cv2.imread('yosemite_meadows.jpg')
img_pencils = cv2.imread('coloured_pencils.jpg')

plot_channels(cv2.cvtColor(img_yosemite, cv2.COLOR_BGR2HSV), "HSV", "Yosemite")
plot_channels(cv2.cvtColor(img_yosemite, cv2.COLOR_BGR2Lab), "Lab", "Yosemite")
plot_channels(cv2.cvtColor(img_pencils,  cv2.COLOR_BGR2HSV), "HSV", "Pencils")
plot_channels(cv2.cvtColor(img_pencils,  cv2.COLOR_BGR2Lab), "Lab", "Pencils")


# b)

def modify_color_channel(path, color_space="HSV", channel=0, factor=0):

    img = cv2.imread(path)

    # Convertir imagen a RGB para mostrar
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convertir la imagen al espacio de color elegido
    if color_space == "HSV":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        back_flag = cv2.COLOR_HSV2RGB
        labels = ["H", "S", "V"]
    else:
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        back_flag = cv2.COLOR_Lab2RGB
        labels = ["L", "a", "b"]

    # Separar canales y modificar el canal deseado
    channels = list(cv2.split(img_cs))
    channels[channel] = (channels[channel] * factor).astype(np.uint8)
    modified = cv2.merge(channels)

    # Convertir de vuelta a RGB para mostrar
    modified_rgb = cv2.cvtColor(modified, back_flag)

    # Mostrar la imagen original y la modificada
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb); plt.axis("off"); plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(modified_rgb); plt.axis("off")
    plt.title(f"{color_space}: {labels[channel]} × {factor}")
    plt.show()


modify_color_channel("yosemite_meadows.jpg", "HSV", 0, 0)
modify_color_channel("yosemite_meadows.jpg", "HSV", 1, 0.5)
modify_color_channel("coloured_pencils.jpg", "Lab", 1, 0)
modify_color_channel("coloured_pencils.jpg", "Lab", 2, 0.3)


# Función para sliders interactivos de ajuste de color
def interactive_color_sliders(path, color_space="HSV"):

    img = cv2.imread(path)

    # Convertir a espacio de color
    if color_space == "HSV":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        back_flag = cv2.COLOR_HSV2BGR
        labels = ["H", "S", "V"]
    else:
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        back_flag = cv2.COLOR_Lab2BGR
        labels = ["L", "a", "b"]

    # Separar canales
    splits = cv2.split(img_cs)
    win = f"{color_space} Control"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Crear sliders para cada canal
    for lab in labels:
        cv2.createTrackbar(lab, win, 100, 100, lambda x: None)

    while True:
        if cv2.getWindowProperty(win, 0) < 0:  # Salir si ventana se cierra
            break

        # Obtener factores de los sliders
        factors = [cv2.getTrackbarPos(l, win) / 100 for l in labels]

        # Aplicar factor a cada canal
        mod = [(splits[i] * factors[i]).astype(np.uint8) for i in range(3)]
        merged = cv2.merge(mod)
        shown = cv2.cvtColor(merged, back_flag)

        # Mostrar resultado
        cv2.imshow(win, shown)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cv2.destroyWindow(win)


interactive_color_sliders("yosemite_meadows.jpg", "HSV")
interactive_color_sliders("coloured_pencils.jpg", "Lab")


####################################
# 2. Manual color-based segmentation
####################################

# a)

def threshold_mask(channel, rng):
    return ((channel >= rng[0]) & (channel <= rng[1])).astype(np.uint8) * 255

def rgb_threshold_masks(frame, r_range, g_range, b_range):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(rgb)

    mR = threshold_mask(R, r_range)
    mG = threshold_mask(G, g_range)
    mB = threshold_mask(B, b_range)

    # Combinar máscaras con AND
    combined = cv2.bitwise_and(mR, cv2.bitwise_and(mG, mB))
    return mR, mG, mB, combined


def apply_mask(frame, mask):
    return cv2.bitwise_and(frame, frame, mask=mask)


# Cargar primer frame del video
cap = cv2.VideoCapture("ladja.avi")
ret, frame = cap.read(); cap.release()

# Definir rangos de color para RGB
r_range = (90, 230)
g_range = (90, 230)
b_range = (0, 50)

# Generar máscaras
mR, mG, mB, mComb = rgb_threshold_masks(frame, r_range, g_range, b_range)
masked_frame = apply_mask(frame, mComb)

# Mostrar resultados RGB
plt.figure(figsize=(12, 10))
titles = ["image", "maskR", "maskG", "maskB", "mask (R∧G∧B)", "image×mask"]
imgs = [
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    mR, mG, mB, mComb,
    cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[i], cmap="gray" if i>0 else None)
    plt.title(titles[i]); plt.axis("off")
plt.show()


# b) Funciones de máscara HSV

def hsv_threshold_masks(frame, h_range, s_range, v_range):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mH = threshold_mask(H, h_range)
    mS = threshold_mask(S, s_range)
    mV = threshold_mask(V, v_range)
    combined = cv2.bitwise_and(mH, cv2.bitwise_and(mS, mV))
    return mH, mS, mV, combined


# Cargar primer frame del video
cap = cv2.VideoCapture("ladja.avi")
ret, frame = cap.read()
cap.release()

# Rango de ejemplo para HSV
h_range = (20, 35)
s_range = (120, 255)
v_range = (120, 255)

# Generar máscaras HSV
mH, mS, mV, mComb = hsv_threshold_masks(frame, h_range, s_range, v_range)
masked_frame = apply_mask(frame, mComb)

# Mostrar resultados HSV
plt.figure(figsize=(12, 10))
titles = ["image", "maskH", "maskS", "maskV", "mask (H∧S∧V)", "image×mask"]
imgs = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mH, mS, mV, mComb, cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imgs[i], cmap="gray" if i > 0 else None)
    plt.title(titles[i])
    plt.axis("off")
plt.show()


# Función para procesar un video y aplicar segmentación HSV
def process_video(input_path, output_path, h_range, s_range, v_range):

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"FMP4")
    w = int(cap.get(3)); h = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = hsv_threshold_masks(frame, h_range, s_range, v_range)
        out.write(cv2.bitwise_and(frame, frame, mask=mask))

    cap.release(); out.release()

'''
#######################################
# 3. Automatic color-based segmentation
#######################################

region_points = []

# Función para seleccionar la region con el mouse
def select_region_cv2(event, x, y, flags, param):

    global region_points
    if event == cv2.EVENT_LBUTTONDOWN:
        region_points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        region_points.append((x, y))
        cv2.rectangle(param, region_points[0], region_points[1], (0, 255, 0), 2)
        cv2.imshow("Select region", param)


# Función para construir histograma 3D RGB
def build_color_histogram(region, n_bins=32):

    hist = np.zeros((n_bins, n_bins, n_bins), dtype=np.float32)
    bin_size = 255 / n_bins

    R_idx = np.floor(region[:, :, 2] / bin_size).astype(int)
    G_idx = np.floor(region[:, :, 1] / bin_size).astype(int)
    B_idx = np.floor(region[:, :, 0] / bin_size).astype(int)

    # Asegurarse de que los índices estén dentro del rango
    R_idx = np.clip(R_idx, 0, n_bins - 1)
    G_idx = np.clip(G_idx, 0, n_bins - 1)
    B_idx = np.clip(B_idx, 0, n_bins - 1)

    np.add.at(hist, (R_idx.flatten(), G_idx.flatten(), B_idx.flatten()), 1)
    hist = (hist / hist.max() * 255).astype(np.uint8)
    return hist


# Función para segmentar imagen usando histograma 3D
def apply_histogram_segmentation(frame, hist, n_bins=32):
   
    bin_size = 255 / n_bins
    R_idx = np.floor(frame[:, :, 2] / bin_size).astype(int)
    G_idx = np.floor(frame[:, :, 1] / bin_size).astype(int)
    B_idx = np.floor(frame[:, :, 0] / bin_size).astype(int)

    # Evitar índices fuera de rango
    R_idx = np.clip(R_idx, 0, n_bins - 1)
    G_idx = np.clip(G_idx, 0, n_bins - 1)
    B_idx = np.clip(B_idx, 0, n_bins - 1)

    segmented = hist[R_idx, G_idx, B_idx]
    return segmented


# Ejecutar segmentación automática
cap = cv2.VideoCapture("ladja.avi")
ret, frame = cap.read()
cap.release()

clone = frame.copy()
cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", select_region_cv2, clone)

while True:
    cv2.imshow("Select Region", clone)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or len(region_points) == 2:  # ESC o ambos puntos seleccionados
        break
cv2.destroyAllWindows()

# Obtener coordenadas de la region
x1, y1 = region_points[0]
x2, y2 = region_points[1]
x1, x2 = min(x1, x2), max(x1, x2)
y1, y2 = min(y1, y2), max(y1, y2)

# Extraer region y construir histograma
region = frame[y1:y2, x1:x2]
hist = build_color_histogram(region, n_bins=32)
segmented_frame = apply_histogram_segmentation(frame, hist, n_bins=32)

# Mostrar resultado de segmentación
plt.figure(figsize=(8, 6))
plt.imshow(segmented_frame, cmap='gray')
plt.title("Histogram-based segmentation")
plt.axis("off")
plt.show()
