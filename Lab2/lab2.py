import cv2
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
import matplotlib


circles_dir = "assignment2_data/1/"
checkerboard_dir = "assignment2_data/2/"



########################
# 2. CAMERA CALIBRATION
########################

def make_checkerboard_objp():
    # Definición de los puntos del patrón de tablero de ajedrez (checkerboard).
    # Se crea una cuadrícula de coordenadas 3D (x, y, z) que representan las
    # esquinas internas del patrón en coordenadas del mundo (z = 0).
    grid_size = (7, 5)
    # Inicializa una matriz de ceros para N puntos con 3 columnas (X, Y, Z).
    grid = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    # Rellena las columnas X e Y con una malla regular; Z queda en 0.
    grid[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    return grid, grid_size


def make_asymmetric_circle_objp():
    # Definición de los puntos para un patrón de círculos asimétrico.
    # El patrón asimétrico desplaza alternadamente las columnas para formar la retícula.
    circle_grid_size = (4, 11)
    # j corresponde a columnas, i a filas
    j, i = np.meshgrid(np.arange(circle_grid_size[0]), np.arange(circle_grid_size[1]))
    # Posición X ajustada para la asimetría (columnas desplazadas en función de la fila)
    x = (2 * j + i % 2)
    y = i
    z = np.zeros_like(x)
    # Se empaquetan las coordenadas en una matriz Nx3 de tipo float32
    grid = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32)
    return grid, circle_grid_size


# a)

def detect_and_save_calibration_points(image_dir, pattern_type='checkerboard', output_file='calibration_points.npy'):
    # Detecta los puntos del patrón (esquinas del checkerboard o centros de círculos)
    # en todas las imágenes del directorio image_dir y guarda los resultados en output_file.

    # Selección del generador de puntos 3D y función de búsqueda según el tipo de patrón
    if pattern_type == 'checkerboard':
        objp, grid_size = make_checkerboard_objp() # puntos teoricos (vida real)
        find_pattern = cv2.findChessboardCorners
        pattern_flags = 0
    elif pattern_type == 'circles':
        objp, grid_size = make_asymmetric_circle_objp()  # puntos teoricos (vida real)
        find_pattern = cv2.findCirclesGrid
        pattern_flags = cv2.CALIB_CB_ASYMMETRIC_GRID
    else:
        raise ValueError("pattern_type debe ser 'checkerboard' o 'circles'")

    objpoints = []       # Lista de arrays de puntos 3D (coordenadas del patrón)
    imgpoints = []       # Lista de arrays de puntos 2D detectados en la imagen
    detected_images = [] # Rutas de las imágenes donde se detectó correctamente el patrón

    # Recolecta todas las imágenes .png del directorio (ordenadas)
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if not image_paths:
        print(f"No se encontraron imágenes en {image_dir}")
        return

    print(f"Detectando patrón {pattern_type} en {len(image_paths)} imágenes...")

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            # Si la imagen no se puede leer, se salta
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Intenta encontrar el patrón en la imagen en escala de grises para reducir el ruido y mejorar el contraste
        ret, corners = find_pattern(gray, grid_size, flags=pattern_flags) # puntos detectados en la imagen
        if not ret:
            # Si no se encuentra el patrón, informa y continúa con la siguiente imagen
            print(f"No se encontró el patrón en {os.path.basename(img_path)}")
            continue

        # Añade los puntos 3D teóricos y los puntos 2D detectados
        objpoints.append(objp)
        imgpoints.append(corners)
        detected_images.append(img_path)

        # Visualización: dibuja esquinas detectadas y muestra la imagen
        vis = img.copy()
        cv2.drawChessboardCorners(vis, grid_size, corners, ret)
        cv2.imshow(f'{pattern_type} detected', vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # Guarda los resultados en un archivo numpy (.npy) para usarlos después en la calibración
    np.save(output_file, {
        'object_points': objpoints,
        'image_points': imgpoints,
        'image_names': detected_images
    })
    print(f"Se guardaron {len(imgpoints)} detecciones en {output_file}")


# Ejecuta la detección para ambos tipos de patrón
detect_and_save_calibration_points(checkerboard_dir, 'checkerboard', 'checkerboard_points.npy')
detect_and_save_calibration_points(circles_dir, 'circles', 'circle_points.npy')


# b)

def perform_camera_calibration(points_file, pattern_type='checkerboard', sample_image=None, output_file='calibration_results.npy'):
    """
    Realiza la calibración de la cámara usando los puntos detectados guardados en los npy.
    Devuelve la matriz de cámara, coeficientes de distorsión y vectores de rotación/traslación.
    """

    data = np.load(points_file, allow_pickle=True).item()
    imgpoints = data['image_points']
    if len(imgpoints) == 0:
        print("No se encontraron detecciones.")
        return

    # Selecciona el patrón teórico 3D según el tipo
    if pattern_type == 'checkerboard':
        objp, _ = make_checkerboard_objp()
    else:
        objp, _ = make_asymmetric_circle_objp()

    # Asociamos el patrón 3D a cada conjunto de puntos 2D detectados
    objpoints = [objp.copy() for _ in imgpoints]

    # Tamaño de imagen
    if sample_image is None:
        sample_image = data['image_names'][0]
    img = cv2.imread(sample_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = (int(gray.shape[1]), int(gray.shape[0]))  # (width, height) como enteros

    # Flags de calibración
    flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL

    # Calibración
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags)

    # Guardar resultados
    np.save(output_file, {
        'ret': ret,
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs
    })

    print(f"Calibración con patrón {pattern_type} completa. Error RMS: {ret:.4f}")

    # Matriz intrínseca
    print("Matriz intrínseca K:")
    print(np.array2string(camera_matrix, formatter={'float_kind': lambda x: f"{x:8.4f}"}))

    # Distorsión
    print("\nCoeficientes de distorsión (k1, k2, p1, p2, k3, k4, k5, k6):")
    print(np.array2string(dist_coeffs.ravel(), formatter={'float_kind': lambda x: f"{x:8.4f}"}))

    # Vectores de rotación
    print("\nVectores de rotación (rvecs):")
    for i, rvec in enumerate(rvecs):
        print(f"Imagen {i + 1}: {np.array2string(rvec.ravel(), formatter={'float_kind': lambda x: f'{x:8.4f}'})}")

    # Vectores de traslación
    print("\nVectores de traslación (tvecs):")
    for i, tvec in enumerate(tvecs):
        print(f"Imagen {i + 1}: {np.array2string(tvec.ravel(), formatter={'float_kind': lambda x: f'{x:8.4f}'})}")

    print("\n")
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


# Ejecuta la calibración para ambos conjuntos de puntos
perform_camera_calibration('checkerboard_points.npy', 'checkerboard', output_file='calibration_results_checkerboard.npy')
perform_camera_calibration('circle_points.npy', 'circles', output_file='calibration_results_circles.npy')



# c)

def reprojection_check(points_file, calib_file, pattern_type='checkerboard'):
    # Verifica la calibración proyectando los puntos 3D (objpoints) a la imagen
    # usando los parámetros calibrados y comparando esas proyecciones con los puntos
    # detectados originalmente (imgpoints). Muestra también una visualización de errores.

    points = np.load(points_file, allow_pickle=True).item()
    calib = np.load(calib_file, allow_pickle=True).item()

    objpoints = points['object_points']
    imgpoints = points['image_points']
    image_names = points['image_names']
    camera_matrix = calib['camera_matrix']
    dist_coeffs = calib['dist_coeffs']
    rvecs = calib['rvecs']
    tvecs = calib['tvecs']

    errors_all = []  # Lista que contendrá vectores de error por imagen

    for i, img_path in enumerate(image_names):
        img = cv2.imread(img_path)
        # Proyecta los puntos 3D a 2D usando la solución calibrada
        imgpts_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        # Calcula el vector de errores (proyección - detección)
        err_vec = imgpts_proj.reshape(-1,2) - imgpoints[i].reshape(-1,2)
        errors_all.append(err_vec)

        # Visualización: dibuja los puntos detectados (rojo) y los reproyectados (verde)
        vis = img.copy()
        for pd, pp in zip(imgpoints[i].reshape(-1,2), imgpts_proj.reshape(-1,2)):
            cv2.circle(vis, tuple(int(x) for x in pd), 5, (0,0,255), -1)  # real (detección)
            cv2.circle(vis, tuple(int(x) for x in pp), 5, (0,255,0), -1)  # proyectado
        cv2.imshow(f"{pattern_type} reprojection", vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Gráfica de dispersion de errores para todas las imágenes
    plt.figure(figsize=(7,7))
    for i, e in enumerate(errors_all):
        plt.scatter(e[:,0], e[:,1], s=20, label=f'{pattern_type} {i+1}')
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.grid(True)
    plt.legend()
    plt.title(f'Errores de reproyección de {pattern_type} (en píxeles)')
    plt.xlabel('Error en X')
    plt.ylabel('Error en Y')
    plt.axis('equal')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()


# Ejecuta la verificación de reproyección para ambos patrones
reprojection_check('checkerboard_points.npy', 'calibration_results_checkerboard.npy', 'checkerboard')
reprojection_check('circle_points.npy', 'calibration_results_circles.npy', 'circles')



# d)


def show_undistortion(points_file, calib_file, pattern_type):
    # Muestra una comparación superpuesta entre la imagen original y la imagen corregida
    # (undistort) usando los parámetros de calibración guardados.
    data = np.load(points_file, allow_pickle=True).item()
    calib = np.load(calib_file, allow_pickle=True).item()
    cam_mtx = calib['camera_matrix']
    dist = calib['dist_coeffs']

    for img_path in data['image_names']:
        img = cv2.imread(img_path)
        # Genera la versión corregida de la imagen (sin distorsión radial)
        und = cv2.undistort(img, cam_mtx, dist)
        # Crea una superposición semitransparente para comparar visualmente
        overlay = cv2.addWeighted(img, 0.5, und, 0.5, 0)
        cv2.imshow(f'{pattern_type} undistort', overlay)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# Visualiza la undistorsión para ambos patrones
show_undistortion('checkerboard_points.npy', 'calibration_results_checkerboard.npy', 'checkerboard')
show_undistortion('circle_points.npy', 'calibration_results_circles.npy', 'circles')



##############################
# 3. PROJECTIVE TRANSFORMATION
##############################

# Carga los parámetros de calibración calculados para el checkerboard
calib_cb = np.load('calibration_results_checkerboard.npy', allow_pickle=True).item()
camera_matrix_cb = calib_cb['camera_matrix']
dist_cb = calib_cb['dist_coeffs']


img_path = 'assignment2_data/2/image_001.png'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"No se pudo leer {img_path}")

# Corrige la distorsión de la imagen usando los parámetros de calibración
undistorted = cv2.undistort(img, camera_matrix_cb, dist_cb)

# Muestra la imagen y permite seleccionar manualmente 4 esquinas internas
matplotlib.use('TkAgg')  # Asegura que matplotlib use un backend interactivo (se necesita para ginput)

plt.close('all')
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
plt.title("Selecciona 4 esquinas internas (en sentido horario desde la esquina superior izquierda)")
plt.axis('on')
plt.tight_layout()

print("Haz clic en 4 puntos en sentido horario desde la esquina superior izquierda...")
plt.show(block=False)

# Espera a que el usuario haga 4 clics; devuelve una lista de puntos [(x,y),...]
src_points = np.array(plt.ginput(4, timeout=0, show_clicks=True), dtype=np.float32)

plt.close('all')
print("Puntos fuente seleccionados:\n", src_points)


# Define los puntos destino (rectángulo ideal) para la rectificación
checker_size = (7, 5)  # número de esquinas internas en x, y
square_size = 50       # tamaño de cada casilla en píxeles (escala arbitraria)
dst_points = np.array([    # las 4 esquinas de un rectángulo. Estas son las coordenadas en la imagen rectificada.
    [0, 0],
    [checker_size[0]*square_size, 0],
    [checker_size[0]*square_size, checker_size[1]*square_size],
    [0, checker_size[1]*square_size]
], dtype=np.float32)

# Calcula la matriz de homografía que transforma src_points -> dst_points
H, _ = cv2.findHomography(src_points, dst_points)
print("Matriz de homografía:\n", H)

# Aplica la homografía para obtener una vista top-down (rectificada) del tablero
warped = cv2.warpPerspective(undistorted, H,(int(checker_size[0]*square_size), int(checker_size[1]*square_size))
)

# Crea una cuadrícula rectangular de posiciones ideales (en coordenadas de la imagen rectificada)
nx, ny = checker_size
x = np.linspace(0, nx*square_size, nx)
y = np.linspace(0, ny*square_size, ny)
xx, yy = np.meshgrid(x, y)
grid_points = np.stack((xx, yy), axis=-1).reshape(-1, 2)

# Visualiza la imagen rectificada y dibuja la cuadrícula generada
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# Puntos de cuadrícula generados (en rojo) — representan las ubicaciones ideales
plt.scatter(grid_points[:, 0], grid_points[:, 1], c='r', s=25, label='Puntos de cuadrícula generados')

# Los 4 puntos destino usados (en verde) — las esquinas del rectángulo destino
plt.scatter(dst_points[:, 0], dst_points[:, 1], c='lime', s=60, label='4 puntos seleccionados')

plt.title('Tablero transformado: cuadrícula roja + esquinas verdes')
plt.legend()
plt.show()


# Mapea la cuadrícula rectificada de vuelta a la imagen original usando la homografía inversa
H_inv = np.linalg.inv(H)
num_points = grid_points.shape[0]
# Convierte a coordenadas homogéneas (x, y, 1)
homog_points = np.hstack((grid_points, np.ones((num_points, 1)))).T
mapped_points = H_inv @ homog_points
# Normaliza por la coordenada homogénea
mapped_points /= mapped_points[2, :]
mapped_points = mapped_points[:2, :].T

# Visualiza los puntos re-proyectados sobre la imagen sin distorsión
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
plt.scatter(mapped_points[:, 0], mapped_points[:, 1], c='lime', s=20)
plt.title('Puntos de cuadrícula proyectados de vuelta en la imagen sin distorsión')
plt.show()

# Guarda los resultados de la homografía y las cuadrículas
np.save('homography_points.npy', {
    'src_points': src_points,
    'dst_points': dst_points,
    'H': H,
    'grid_points_rectified': grid_points,
    'grid_points_original': mapped_points
})
print("Resultados de homografía guardados en homography_points.npy")




######################
# 4. RADIAL DISTORTION
######################

# a)

# Funciones matemáticas para distorsión radial y su inversa
def distort_radius(r, f):
    """Aplica la función de distorsión radial al radio r con parámetro f.
    La fórmula usada: r' = f * ln(r/f + sqrt(1 + (r/f)^2))
    """
    return f * np.log(r / f + np.sqrt(1 + (r / f)**2))

def undistort_radius(rp, f):
    """Función inversa analítica de la distorsión radial.
    Dada la r' (radio distorsionado) devuelve r (radio original).
    Fórmula derivada analíticamente de la invertida de distort_radius.
    """
    return - (f / 2) * ((np.exp(-2 * rp / f) - 1) / np.exp(-rp / f))

# Dimensiones de la imagen simulada en píxeles (para visualizar la distorsión)
img_width, img_height = 400, 300

# Genera una cuadrícula de puntos sobre la imagen de prueba
grid_size_x, grid_size_y = 15, 11  # cuadrícula relativamente densa
x = np.linspace(0, img_width, grid_size_x)
y = np.linspace(0, img_height, grid_size_y)
xx, yy = np.meshgrid(x, y)
points = np.stack((xx, yy), axis=-1).reshape(-1, 2)

# Centra la cuadrícula en el origen (0,0) para aplicar la distorsión radial correctamente
center = np.array([img_width / 2, img_height / 2])
points_centered = points - center

# Convierte las coordenadas Cartesianas centradas a polares (r, theta)
x_c, y_c = points_centered[:, 0], points_centered[:, 1]
r = np.sqrt(x_c**2 + y_c**2)
theta = np.arctan2(y_c, x_c)

# Parámetro de distorsión f (controla la intensidad de la distorsión)
f = 200

# Aplica la distorsión radial en el dominio del radio
r_distorted = distort_radius(r, f)
x_distorted = r_distorted * np.cos(theta)
y_distorted = r_distorted * np.sin(theta)

# Traslada de nuevo al sistema de coordenadas de la imagen (centro)
points_distorted = np.stack((x_distorted, y_distorted), axis=-1) + center

# Aplica la función inversa para intentar recuperar los puntos originales
r_restored = undistort_radius(r_distorted, f)
x_restored = r_restored * np.cos(theta)
y_restored = r_restored * np.sin(theta)
points_restored = np.stack((x_restored, y_restored), axis=-1) + center

# Visualiza la cuadrícula original, distorsionada y restaurada
plt.figure(figsize=(12, 10))

# Puntos distorsionados (azul)
plt.scatter(points_distorted[:, 0], points_distorted[:, 1], c='blue', s=25, alpha=0.7, label='Distorsionados')

# Puntos restaurados tras aplicar la inversa (magenta)
plt.scatter(points_restored[:, 0], points_restored[:, 1], c='magenta', s=25, alpha=0.7, label='Restaurados (inversa)')

# Puntos originales (rojo), se dibujan encima para comparar
plt.scatter(points[:, 0], points[:, 1], c='red', s=50, alpha=0.9, label='Originales')

# Línea central (ejes) para referencia visual
plt.axhline(img_height / 2, color='k', linestyle='--')
plt.axvline(img_width / 2, color='k', linestyle='--')

plt.title(f'Visualización de distorsión radial (f={f})')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlim(0, img_width)
plt.ylim(0, img_height)
plt.show()


# b)

img_path = "assignment2_data/2/image_001.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"No se pudo abrir {img_path}")

# Convierte a RGB para mostrar con matplotlib (OpenCV usa BGR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
center = np.array([w / 2, h / 2])

# función para "undistort" de puntos usando el parámetro f y el modelo analítico
def undistort_points(points, f, center):
    # Restaura los puntos centrados -> calcula r' (radio distorsionado) -> aplica la inversa analítica.
    points_centered = points - center
    x, y = points_centered[:, 0], points_centered[:, 1]
    r_p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # Aplicamos la fórmula analítica inversa (igual que undistort_radius)
    r = - (f / 2) * ((np.exp(-2 * r_p / f) - 1) / np.exp(-r_p / f))
    x_new = r * np.cos(theta)
    y_new = r * np.sin(theta)
    return np.stack((x_new, y_new), axis=-1) + center

# Paso 1: seleccionar manualmente una serie de puntos a lo largo de una curva visible
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title("Selecciona puntos a lo largo de una curva cerca del extremo de la imagen.\nPresiona Enter cuando termines.")
plt.axis('on')
selected_points = np.array(plt.ginput(n=-1, timeout=0), dtype=np.float32)  # n=-1 permite seleccionar indefinidamente hasta Enter
plt.close('all')

if len(selected_points) < 3:
    raise ValueError("¡Selecciona al menos 3 puntos!")

x = selected_points[:, 0]
y = selected_points[:, 1]

# Paso 2: ajustar una recta (polinomio de grado 1) a los puntos originales
coef = np.polyfit(x, y, 1)
poly_fn = np.poly1d(coef)
y_fit = poly_fn(x)

# Paso 3: medir el error promedio vertical absoluto respecto a la recta ajustada
error_original = np.mean(np.abs(y - y_fit))
print(f"Error promedio para la línea original (distorsionada): {error_original:.4f} px")

# Paso 4: barrido (grid search) sobre candidatos de f para minimizar el error después de undistort
f_candidates = np.linspace(200, 800, 13)  # rango de búsqueda para f
errors = []

for f_candidate in f_candidates:
    pts_undist = undistort_points(selected_points, f_candidate, center)
    x_u, y_u = pts_undist[:, 0], pts_undist[:, 1]
    # Ajusta una recta a los puntos corregidos
    coef_u = np.polyfit(x_u, y_u, 1)
    poly_fn_u = np.poly1d(coef_u)
    y_fit_u = poly_fn_u(x_u)
    # Calcula el error medio vertical absoluto entre puntos undistorted y la recta ajustada
    err = np.mean(np.abs(y_u - y_fit_u))
    errors.append(err)

# selecciona el f que da el menor error
best_idx = np.argmin(errors)
f_opt = f_candidates[best_idx]
print(f"\nParámetro f óptimo = {f_opt:.2f} (error mínimo = {errors[best_idx]:.4f})")

# visualización del resultado con el f óptimo
pts_best = undistort_points(selected_points, f_opt, center)
x_b, y_b = pts_best[:, 0], pts_best[:, 1]
coef_b = np.polyfit(x_b, y_b, 1)
poly_fn_b = np.poly1d(coef_b)

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
# Puntos originalmente seleccionados (distorsionados)
plt.scatter(selected_points[:, 0], selected_points[:, 1], c='red', s=60, label='Puntos originales seleccionados')
# Línea ajustada a los puntos originales
plt.plot(x, poly_fn(x), 'r--', label='Recta ajustada (original)')
# Puntos después de aplicar la corrección con f óptimo
plt.scatter(x_b, y_b, c='cyan', s=60, label='Puntos undistorted (óptimo)')
# Línea ajustada a los puntos corregidos
plt.plot(x_b, poly_fn_b(x_b), 'c-', label=f'Recta ajustada (undistorted, f={f_opt:.0f})')
plt.title(f'Prueba de enderezamiento de línea (f óptimo = {f_opt:.0f})')
plt.legend()
plt.axis('equal')
plt.show()
