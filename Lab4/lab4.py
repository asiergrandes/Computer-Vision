import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Sequence

# Edge detection
img_name = "building.jpg"
sobel_threshold = 100
canny_threshold_low = 50
canny_threshold_high = 150

# Video
video = 'squash.avi'
frames = 10
background = 'background_median.jpg'
mask_threshold = 30
player_area = 500

# Players
hist_threshold = 0.5
blue_player = (255, 0, 0)   # BGR
red_player = (0, 0, 255)    # BGR


###################
# 1. EDGE DETECTION
###################

# Función: sobel_edges
# Descripción: Calcula las componentes del gradiente en X e Y usando el kernel de Sobel
# (mediante convolución), la magnitud del gradiente en punto flotante, una versión
# normalizada a uint8 para visualización y una imagen binaria umbralizada basada en
# la magnitud (255 donde la magnitud > threshold, 0 en caso contrario).
# Entradas:
#   - img_gray: imagen en escala de grises (uint8)
#   - threshold: umbral aplicado sobre la magnitud (float)
# Salidas (tupla): (Gx, Gy, G, G_norm, binary)
def sobel_edges(img_gray: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Sobel Gx, Gy, gradient magnitude (float32) and normalized uint8 magnitude.
    Also returns binary thresholded image (uint8 0/255) using provided threshold on the float magnitude.
    """
    img_f = img_gray.astype(np.float32)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    Gx = cv2.filter2D(img_f, ddepth=cv2.CV_32F, kernel=kx)
    Gy = cv2.filter2D(img_f, ddepth=cv2.CV_32F, kernel=ky)
    G = np.sqrt(Gx**2 + Gy**2)

    G_norm = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    binary = np.where(G > float(threshold), 255, 0).astype(np.uint8)

    return Gx, Gy, G, G_norm, binary


# Función: canny_edges
# Descripción: Aplica el detector de bordes de Canny y devuelve el mapa de bordes binario.
# Entradas: imagen en escala de grises y los umbrales low/high para Canny.
def canny_edges(img_gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(img_gray, low, high)


# Función: plot_edge_results
# Descripción: Muestra en una figura 2x3 los resultados intermedios del análisis de bordes:
# imagen original en gris, respuestas Gx/Gy, magnitud normalizada, mapa binario sobel y resultado de Canny.
# No altera datos, solo visualiza para debug/inspección.
def plot_edge_results(img_gray: np.ndarray, Gx: np.ndarray, Gy: np.ndarray, G_norm: np.ndarray,
                      sobel_binary: np.ndarray, canny_map: np.ndarray, sobel_thresh: int,
                      canny_low: int, canny_high: int, title_name: str = "hotel.jpg") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Edge Detection Results for {title_name}", fontsize=16)

    axes[0, 0].imshow(img_gray, cmap='gray'); axes[0, 0].set_title('A) Original Grayscale'); axes[0, 0].axis('off')
    axes[0, 1].imshow(Gx, cmap='gray'); axes[0, 1].set_title(r'$G_x$ (X-axis Response)'); axes[0, 1].axis('off')
    axes[0, 2].imshow(Gy, cmap='gray'); axes[0, 2].set_title(r'$G_y$ (Y-axis Response)'); axes[0, 2].axis('off')
    axes[1, 0].imshow(G_norm, cmap='gray'); axes[1, 0].set_title(r'B) Gradient Magnitude $G$'); axes[1, 0].axis('off')

    axes[1, 1].imshow(sobel_binary, cmap='gray')
    axes[1, 1].set_title(f'C) Sobel Binary Edges (Threshold={sobel_thresh})')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(canny_map, cmap='gray')
    axes[1, 2].set_title(f'D) Canny Edges (Low={canny_low}, High={canny_high})')
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


####################
# 2. HOUGH TRANSFORM
####################

# Función: hough_accumulator_from_edges
# Descripción: Dado un mapa de bordes binario, construye un acumulador A(k_idx, n_idx)
# para la representación de rectas y = k*x + n. Recorre una discretización de k y,
# para cada k computa n = y - k*x y acumula en los bins correspondientes.
# Retorna la matriz acumuladora A y los vectores ks y ns que definen los ejes.
def hough_accumulator_from_edges(edges: np.ndarray,
                                 k_range: Tuple[float, float] = (-2.0, 2.0),
                                 k_bins: int = 400,
                                 n_range: Tuple[float, float] = (-200.0, 200.0),
                                 n_bins: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edges.ndim != 2:
        raise ValueError("edges must be a 2D array")

    ks = np.linspace(k_range[0], k_range[1], k_bins, dtype=np.float32)
    ns = np.linspace(n_range[0], n_range[1], n_bins, dtype=np.float32)
    A = np.zeros((k_bins, n_bins), dtype=np.int32)

    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return A, ks, ns

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    n_bin_size = (n_range[1] - n_range[0]) / (n_bins - 1) if n_bins > 1 else 1.0

    for k_idx, k in enumerate(ks):
        n_vals = ys_f - k * xs_f
        idx_f = (n_vals - n_range[0]) / n_bin_size
        # keep only valid before rounding (avoid out-of-bounds)
        valid_mask = (idx_f >= 0) & (idx_f < n_bins)
        if not np.any(valid_mask):
            continue
        idxs = np.rint(idx_f[valid_mask]).astype(np.int32)
        # final safety clipping
        idxs = idxs[(idxs >= 0) & (idxs < n_bins)]
        if idxs.size == 0:
            continue
        # aggregate counts per bin to speed up repeated indices
        counts = np.bincount(idxs, minlength=n_bins)
        A[k_idx] += counts

    return A, ks, ns


# Función: hough_peaks
# Descripción: Extrae los picos más fuertes del acumulador A. Implementa supresión no máxima
# por ventana (nms_size) poniendo a cero la región alrededor de cada pico encontrado. Devuelve
# una lista de picos como (k_idx, n_idx, votos).
# Además dibuja una visualización simple del acumulador y marca el último bloque de nms usado
# (para inspección). Nota: la visualización no altera los datos devueltos.
def hough_peaks(A: np.ndarray, num_peaks: int = 10, nms_size: Tuple[int, int] = (9, 9)) -> List[Tuple[int, int, int]]:
    A_work = A.copy()
    k_bins, n_bins = A_work.shape
    peaks = []

    for _ in range(num_peaks):
        idx_flat = int(A_work.argmax())
        votes = int(A_work.flat[idx_flat])
        if votes == 0:
            break
        k_idx, n_idx = np.unravel_index(idx_flat, A_work.shape)
        peaks.append((int(k_idx), int(n_idx), votes))
        k_w, n_w = nms_size
        k_half, n_half = k_w // 2, n_w // 2
        k0 = max(0, k_idx - k_half); k1 = min(k_bins, k_idx + k_half + 1)
        n0 = max(0, n_idx - n_half); n1 = min(n_bins, n_idx + n_half + 1)
        A_work[k0:k1, n0:n1] = 0

    # Create a new figure for the accumulator visualization
    plt.figure(figsize=(6, 6))

    # Show the accumulator matrix as an image
    plt.imshow(A, cmap="gray", aspect="auto")

    # Overlay the top N peaks as red points
    plt.scatter(n0, k0, s=40, facecolors="none", edgecolors="red")

    # Label the axes to indicate which dimension corresponds to k and n
    plt.xlabel("n index")
    plt.ylabel("k index")
    plt.title("Accumulator with top peak")

    # Show the accumulator figure
    plt.show()
    return peaks


# Función: peaks_to_lines
# Descripción: Convierte los índices de picos (k_idx, n_idx) en coordenadas de segmentos de línea
# dentro del tamaño de imagen dado. Calcula intersecciones con los bordes del rectángulo de la
# imagen para generar dos puntos por línea en coordenadas enteras (x1,y1,x2,y2).
def peaks_to_lines(peaks: Sequence[Tuple[int, int, int]], ks: np.ndarray, ns: np.ndarray,
                   img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    h, w = img_shape[0], img_shape[1]
    lines = []

    for (k_idx, n_idx, _) in peaks:
        k = float(ks[int(k_idx)])
        n = float(ns[int(n_idx)])

        x0, x1 = 0.0, float(w - 1)
        y0, y1 = k * x0 + n, k * x1 + n

        pts = []

        def add_if_in(x, y):
            if 0.0 <= x <= (w - 1) + 1e-6 and 0.0 <= y <= (h - 1) + 1e-6:
                pts.append((int(round(x)), int(round(y))))

        add_if_in(x0, y0); add_if_in(x1, y1)

        if len(pts) < 2:
            if abs(k) > 1e-8:
                for y_border in (0.0, float(h - 1)):
                    x_at_y = (y_border - n) / k
                    add_if_in(x_at_y, y_border)
            else:
                if 0.0 <= n <= (h - 1):
                    add_if_in(0.0, n); add_if_in(float(w - 1), n)

        if len(pts) < 2:
            continue

        unique_pts = []
        for p in pts:
            if p not in unique_pts:
                unique_pts.append(p)
            if len(unique_pts) >= 2:
                break
        if len(unique_pts) >= 2:
            x1i, y1i = unique_pts[0]; x2i, y2i = unique_pts[1]
            lines.append((x1i, y1i, x2i, y2i))

    return lines


# Función: draw_lines_on_image
# Descripción: Dibuja los segmentos de línea (lista de tuplas x1,y1,x2,y2) sobre una copia de la imagen BGR
# usando cv2.line y devuelve la imagen resultante. Parámetros opcionales: color y grosor.
def draw_lines_on_image(img_bgr: np.ndarray, lines: Sequence[Tuple[int, int, int, int]],
                        color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in lines:
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    return out


# Función: plot_lines_on_image
# Descripción: Convierte la imagen con líneas a RGB y la muestra con matplotlib (sin ejes).
# Útil para visualizar el resultado en notebooks o scripts con interfaz.
def plot_lines_on_image(img_bgr: np.ndarray, lines: Sequence[Tuple[int, int, int, int]], figsize: Tuple[int, int] = (12, 8)):
    img_with = draw_lines_on_image(img_bgr, lines)
    img_rgb = cv2.cvtColor(img_with, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb); plt.axis('off'); plt.show()




##########################
# 3. BACKGROUND EXTRACTION
##########################

# Función: calculate_background
# Descripción: Abre un video, muestrea frames cada 'step' frames, y calcula la media y la mediana
# por pixel a lo largo de los frames muestreados. Devuelve ambas imágenes de fondo (mean, median)
# como arrays uint8. Si no puede abrir el video o no hay frames muestreados, retorna (None, None).
def calculate_background(video_path: str, step: int) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    sampled = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            sampled.append(frame)
        frame_count += 1
    cap.release()

    if not sampled:
        print("No frames sampled.")
        return None, None

    frames_array = np.array(sampled, dtype=np.uint8)
    mean_background = np.mean(frames_array, axis=0).astype(np.uint8)
    median_background = np.median(frames_array, axis=0).astype(np.uint8)

    return mean_background, median_background


############################
# 4. PLAYER REGION DETECTION
############################

# Función: euclidean_distance_diff
# Descripción: Calcula la distancia euclidiana (L2) por píxel entre dos imágenes color (I1, I2).
# Produce un mapa escalar (por píxel) con la magnitud de la diferencia, lo normaliza a 0..255
# y lo devuelve como uint8. Se usa para detectar movimiento / foreground respecto al fondo.
def euclidean_distance_diff(I1: np.ndarray, I2: np.ndarray) -> np.ndarray:
    I1_float = I1.astype(np.float32); I2_float = I2.astype(np.float32)
    diff_sq = (I1_float - I2_float) ** 2
    sum_diff_sq = np.sum(diff_sq, axis=2)
    L2_diff = np.sqrt(sum_diff_sq)
    L2_diff_norm = cv2.normalize(L2_diff, None, 0, 255, cv2.NORM_MINMAX)
    return L2_diff_norm.astype(np.uint8)


# Función: display_intermediate_results
# Descripción: Muestra una figura 2x2 con resultados intermedios del detector de jugadores: el frame coloreado,
# el mapa L2 de diferencias, la máscara binaria inicial (aplicando mask_threshold) y la máscara limpiada
# tras operaciones morfológicas. Es solo visualización para debugging/inspección.
def display_intermediate_results(frame: np.ndarray, diff_L2: np.ndarray, cleaned_mask: np.ndarray, diff_abs: np.ndarray):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Intermediate Player Detection Results (Single Frame)", fontsize=14)

    axes[0, 0].imshow(frame_rgb); axes[0, 0].set_title('A) Frame with Player Bounding Boxes'); axes[0, 0].axis('off')
    axes[0, 1].imshow(diff_L2, cmap='gray'); axes[0, 1].set_title(r'B) $L_2$ (Euclidean) Difference'); axes[0, 1].axis('off')

    _, thresh_initial = cv2.threshold(diff_L2, mask_threshold, 255, cv2.THRESH_BINARY)
    axes[1, 0].imshow(thresh_initial, cmap='gray'); axes[1, 0].set_title(f'C) Initial Binary Mask (Threshold={mask_threshold})'); axes[1, 0].axis('off')

    axes[1, 1].imshow(cleaned_mask, cmap='gray'); axes[1, 1].set_title('D) Cleaned Morphological Mask'); axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


##########################
# 5. PLAYER IDENTIFICATION
##########################

# Función: calculate_histogram
# Descripción: Calcula un histograma 2D de los canales H y S en espacio HSV para la ROI dada,
# lo normaliza a rango 0..1 y lo devuelve. Se usa para caracterizar el color del jugador.
def calculate_histogram(image_roi: np.ndarray):
    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# Función: extract_reference_histograms
# Descripción: Extrae de manera "segura" dos histogramas de referencia desde el primer frame
# del video. Define dos ROIs (estáticas) para jugador1 y jugador2, las recorta y calcula los
# histogramas correspondientes. Devuelve (hist_p1, hist_p2, frame) o (None, None, None)
# si ocurre un fallo (archivo no encontrado, frame inválido o ROI fuera de rango).
def extract_reference_histograms(video_path, show_rois=False):
    """
    Safely extract two reference histograms from the first frame.
    Returns (hist_p1, hist_p2, frame) or (None, None, None) on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Error: Could not read first frame from video.")
        return None, None, None

    h_frame, w_frame = frame.shape[:2]
    print("First frame size (w,h):", w_frame, h_frame)

    roi_p1 = (100, 150, 80, 150)   # (x, y, w, h)
    roi_p2 = (250, 100, 80, 150)

    def clamp_roi(roi):
        x, y, w, h = roi
        x = max(0, int(x)); y = max(0, int(y))
        w = max(0, int(w)); h = max(0, int(h))
        # clamp width/height so ROI is inside the frame
        if x + w > w_frame:
            w = max(0, w_frame - x)
        if y + h > h_frame:
            h = max(0, h_frame - y)
        return x, y, w, h

    x1, y1, w1, h1 = clamp_roi(roi_p1)
    x2, y2, w2, h2 = clamp_roi(roi_p2)

    if w1 == 0 or h1 == 0:
        print("Warning: ROI1 is empty after clamping. Check roi_p1 values:", roi_p1)
        return None, None, frame
    if w2 == 0 or h2 == 0:
        print("Warning: ROI2 is empty after clamping. Check roi_p2 values:", roi_p2)
        return None, None, frame

    roi1 = frame[y1:y1+h1, x1:x1+w1]
    roi2 = frame[y2:y2+h2, x2:x2+w2]

    # Optional visualization to confirm ROIs
    if show_rois:
        disp = frame.copy()
        cv2.rectangle(disp, (x1,y1), (x1+w1, y1+h1), (0,255,0), 2)
        cv2.rectangle(disp, (x2,y2), (x2+w2, y2+h2), (0,0,255), 2)
        cv2.imshow("First frame ROIs", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    hist_p1 = calculate_histogram(roi1)
    hist_p2 = calculate_histogram(roi2)

    print("Reference Histograms extracted successfully.")
    return hist_p1, hist_p2, frame


# Función: identify_and_track_players
# Descripción: Recorre el video, resta el fondo por diferencia L2, obtiene una máscara binaria,
# aplica operaciones morfológicas, encuentra componentes conectadas grandes (posibles jugadores),
# calcula histogramas de cada ROI detectada y compara con los histogramas de referencia usando
# la distancia de Bhattacharyya para asignar etiquetas/colores a los jugadores. Escribe un video
# de salida con las cajas y etiquetas dibujadas.
def identify_and_track_players(video_path: str, bg_path: str, ref_hist_p1, ref_hist_p2):
    bg_image = cv2.imread(bg_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('player_identification_output.mp4', fourcc, fps, (frame_width, frame_height))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("Processing video frames and identifying players...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        diff_image_L2 = euclidean_distance_diff(frame, bg_image)
        _, thresh_mask = cv2.threshold(diff_image_L2, mask_threshold, 255, cv2.THRESH_BINARY)
        mask_open = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, 8, cv2.CV_32S)
        component_areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
        component_areas.sort(key=lambda x: x[0], reverse=True)

        for i in range(min(2, len(component_areas))):
            area, label_index = component_areas[i]
            if area > player_area:
                x = stats[label_index, cv2.CC_STAT_LEFT]; y = stats[label_index, cv2.CC_STAT_TOP]
                w = stats[label_index, cv2.CC_STAT_WIDTH]; h = stats[label_index, cv2.CC_STAT_HEIGHT]
                cx, cy = int(centroids[label_index, 0]), int(centroids[label_index, 1])

                current_roi = frame[y:y + h, x:x + w]
                if current_roi.size == 0:
                    continue
                current_hist = calculate_histogram(current_roi)

                d1 = cv2.compareHist(ref_hist_p1, current_hist, cv2.HISTCMP_BHATTACHARYYA)
                d2 = cv2.compareHist(ref_hist_p2, current_hist, cv2.HISTCMP_BHATTACHARYYA)

                if d1 < d2 and d1 < hist_threshold:
                    player_id = 1
                    color = blue_player
                elif d2 < d1 and d2 < hist_threshold:
                    player_id = 0
                    color = (255,255,255)
                else:
                    player_id = 2;
                    color = red_player

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.circle(frame, (cx, cy), 5, color, -1)

                label = f"P{player_id}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)

    cap.release(); out.release()
    print("\nPlayer identification complete. Output video saved as 'player_identification_output.mp4'.")


# Función: to_gray_uint8
# Descripción: Conversión simple de BGR a escala de grises uint8 usando OpenCV.
def to_gray_uint8(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# EDGE DETECTION CALLS

try:
    img_color = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img_color is not None:
        img_gray = to_gray_uint8(img_color)
        Gx, Gy, G, G_display, binary_sobel = sobel_edges(img_gray, sobel_threshold)
        canny_map = canny_edges(img_gray, canny_threshold_low, canny_threshold_high)
        # display (matches the previous plotting)
        plot_edge_results(img_gray, Gx, Gy, G_display, binary_sobel, canny_map,
                          sobel_threshold, canny_threshold_low, canny_threshold_high, title_name=img_name)
    else:
        print(f"Edge demo: '{img_name}' not found — skipping edge demo.")
except Exception as e:
    print("Edge demo failed:", e)


# HOUGH CALL

try:
    img = cv2.imread("hotel.jpg")
    if img is None:
        print("HOUGH demo: 'hotel.jpg' not found — skipping Hough demo.")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 50, 150)
        A, ks, ns = hough_accumulator_from_edges(edges,
                                                 k_range=(-2.0, 2.0), k_bins=400,
                                                 n_range=(-200.0, 200.0), n_bins=400)
        peaks = hough_peaks(A, num_peaks=10, nms_size=(9, 9))
        lines = peaks_to_lines(peaks, ks, ns, img_shape=edges.shape)

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_with_lines = draw_lines_on_image(edges_bgr, lines)

        cv2.imwrite("hotel_lines.png", edges_with_lines)
        plot_lines_on_image(edges_bgr, lines)
except Exception as e:
    print("Hough demo failed:", e)


# BACKGROUND EXTRACTION CALLS

mean_bg, median_bg = calculate_background(video, frames)
if mean_bg is not None and median_bg is not None:
    cv2.imwrite('background_mean.jpg', mean_bg)
    cv2.imwrite('background_median.jpg', median_bg)
    mean_bg_rgb = cv2.cvtColor(mean_bg, cv2.COLOR_BGR2RGB)
    median_bg_rgb = cv2.cvtColor(median_bg, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mean_bg_rgb); axes[0].set_title('Mean Background Image'); axes[0].axis('off')
    axes[1].imshow(median_bg_rgb); axes[1].set_title('Median Background Image'); axes[1].axis('off')
    plt.show()
    print("\nSaved 'background_mean.jpg' and 'background_median.jpg'.")
else:
    print("Background calculation failed — check your video file path.")


# PLAYER REGION DETECTION

bg_image = cv2.imread('background_median.jpg')
cap = cv2.VideoCapture('squash.avi')
ret, frame = cap.read()  # read first frame
cap.release()

if not ret:
    raise ValueError("Could not read a frame from the video")

diff_L2 = euclidean_distance_diff(frame, bg_image)

_, thresh_mask = cv2.threshold(diff_L2, mask_threshold, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

display_intermediate_results(frame, diff_L2, cleaned_mask, diff_L2)


# PLAYER IDENTIFICATION CALL

ref_hist_p1, ref_hist_p2, first_frame = extract_reference_histograms(video)
if ref_hist_p1 is not None and ref_hist_p2 is not None and os.path.exists(background):
    identify_and_track_players(video, background, ref_hist_p1, ref_hist_p2)
else:
    print("Player identification skipped: could not extract reference histograms or background missing.")
