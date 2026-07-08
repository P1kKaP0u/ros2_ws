from datetime import datetime
import numpy as np
import cv2
from skimage.morphology import skeletonize


def pipeBinaryControl(mask, center, minDia=0):
    """
    Mask üzerinde center noktasına göre boru çapını arar.
    mask: binary pipe maskesi
    center: (x,y) koordinatı
    minDia: minimum çap için eşik
    """
    s = mask.shape[1] // 2
    x, y = s, int(center[1])
    yControlpx = y
    a, b = 0, mask.shape[1]

    while b - a > 2:
        # Sol ve sağ striplerde çap kontrolü
        profile_strip = mask[y - yControlpx:y + yControlpx, a:(a + b) // 2]
        thicknesses = np.sum(profile_strip == 255, axis=1)
        thicknesses = thicknesses[thicknesses > 0]
        diameter_px1 = np.median(thicknesses) if len(thicknesses) > minDia else 0

        profile_strip = mask[y - yControlpx:y + yControlpx, (a + b) // 2:b]
        thicknesses = np.sum(profile_strip == 255, axis=1)
        thicknesses = thicknesses[thicknesses > 0]
        diameter_px2 = np.median(thicknesses) if len(thicknesses) > minDia else 0

        if diameter_px2 >= diameter_px1:
            a = (a + b) // 2
        elif diameter_px2 < diameter_px1:
            b = (a + b) // 2
        else:
            break

    profile_strip = mask[y - yControlpx:y + yControlpx, b - 3:b]
    thicknesses = np.sum(profile_strip == 255, axis=0)
    thicknesses = thicknesses[thicknesses > 0]
    diameter_px2 = np.median(thicknesses) if len(thicknesses) > minDia else 0

    return diameter_px2, b


def findDiameters(maskPipe, y):
    """
    Boru maskesinde y konumunda çap ve açı bilgisi hesaplar.
    maskPipe: pipe maskesi (binary)
    y: analiz yapılacak y koordinatı
    """
    contours, _ = cv2.findContours(maskPipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bottomY = 0
    upperY = float('inf')
    diameters = [0, 0]
    angleMain = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            continue

        rect = cv2.minAreaRect(cnt)
        (x2, y2), (w2, h2), angle2 = rect

        if angle2 > 90:
            angle2 += 90

        # Alt ve üst boru konturları arasında seçim
        if bottomY < y2 < y:
            bottomY = y2
            Up = False
            angleMain = angle2
        elif upperY > y2 > y:
            upperY = y2
            Up = True
            angle2 = (angle2 + 180) % 360
        else:
            continue

        # İlgili konturu maskeye çiz
        maskPipe_temp = np.zeros_like(maskPipe)
        cv2.drawContours(maskPipe_temp, [cnt], -1, 255, thickness=-1)

        box = cv2.boxPoints(rect)
        (h, w) = maskPipe_temp.shape[:2]
        center = (int(x2), int(y2))
        M = cv2.getRotationMatrix2D(center, angle2, 1.0)
        rotated_mask = cv2.warpAffine(maskPipe_temp, M, (w, h), flags=cv2.INTER_NEAREST)

        diameter_px, _ = pipeBinaryControl(rotated_mask, center)
        if Up:
            diameters[1] = diameter_px
        else:
            diameters[0] = diameter_px

    return diameters, angleMain


def warp_and_expand_perspective(img, pts1, pts2):
    """
    Perspektif dönüşümü uygular ve görüntüyü sınırlar dışına taşarsa genişletir.
    """
    M = cv2.getPerspectiveTransform(pts1, pts2)
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners[None, :, :], M)[0]

    x_min, x_max = transformed_corners[:, 0].min(), transformed_corners[:, 0].max()
    y_min, y_max = transformed_corners[:, 1].min(), transformed_corners[:, 1].max()

    new_w = int(np.ceil(x_max - x_min))
    new_h = int(np.ceil(y_max - y_min))

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    M_new = translation @ M  # Matris çarpımı

    warped = cv2.warpPerspective(img, M_new, (new_w, new_h), flags=cv2.INTER_NEAREST)
    return warped, new_w, new_h


def rotate_and_expand(image, angle, border_value=0):
    """
    Görüntüyü açısal döndürür, taşan kısımlar için canvas genişletir.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=border_value)
    return rotated


def resize(imgAnomaly, maskPipe, YoverX, y, debug=False):
    """
    Anomali maskesini boru boyutları ve açılarına göre yeniden şekillendirir.
    """
    diameters, angleMain = findDiameters(maskPipe, y)
    ratio = diameters[1] / diameters[0]

    (h, w) = imgAnomaly.shape[:2]
    center = (w / 2, h / 2)

    # Döndür ve genişlet
    rotated_mask = rotate_and_expand(imgAnomaly, angleMain)
    h, w = rotated_mask.shape[:2]
    center = (w / 2, h / 2)

    ratio2 = YoverX * w * ratio / h

    ul = [int(center[0] + w / 2 * ratio), int(center[1] - h / 2)]
    ur = [int(center[0] - w / 2 * ratio), int(center[1] - h / 2)]
    bl = [int(center[0] + w / 2), int(center[1] + h / 2)]
    br = [int(center[0] - w / 2), int(center[1] + h / 2)]

    pts1 = np.float32([ul, ur, bl, br])
    pts2 = np.float32([[int(center[0] + w / 2), int(center[1] - h / 2)],
                       [int(center[0] - w / 2), int(center[1] - h / 2)], bl, br])

    new_img, h, w = warp_and_expand_perspective(rotated_mask, pts1, pts2)

    # İkinci perspektif düzeltme
    center = (w / 2, h / 2)
    ul = [int(center[0] + w / 2), int(center[1] - h / 2 * ratio2)]
    ur = [int(center[0] - w / 2), int(center[1] - h / 2 * ratio2)]
    bl = [int(center[0] + w / 2), int(center[1] + h / 2)]
    br = [int(center[0] - w / 2), int(center[1] + h / 2)]

    pts1 = np.float32([ul, ur, bl, br])
    pts2 = np.float32([[int(center[0] + w / 2), int(center[1] - h / 2)],
                       [int(center[0] - w / 2), int(center[1] - h / 2)], bl, br])

    new_img, h, w = warp_and_expand_perspective(new_img, pts1, pts2)

    if debug:
        cv2.imshow("Resized Anomaly", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return new_img


def get_largest_contour(imgAnomaly, maskPipe, YoverX, y, func=False):
    """
    Anomali maskesinin en büyük konturunu döner.
    """
    if func:
        image = resize(imgAnomaly, maskPipe, YoverX, y)
    else:
        image = imgAnomaly

    if image is None:
        raise ValueError("Görüntü yüklenemedi")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def get_hu_moments(contour):
    """
    Konturun Hu Moment değerlerini döner.
    """
    return cv2.HuMoments(cv2.moments(contour)).flatten()


def match_anomaly(imgAnomaly, maskPipe, YoverX, y):
    """
    Anomali maskesi ile referans şekillerin karşılaştırması,
    en uygun şekil ismini döner.
    """
    reference_shapes = {
        "ucgen": "templates/ucgen.png",
        "kare": "templates/kare.png",
        "dikdortgen": "templates/dikdortgen.png",
    }

    target_contour = get_largest_contour(imgAnomaly, maskPipe, YoverX, y, func=True)
    if target_contour is None:
        return None

    target_hu = get_hu_moments(target_contour)
    best_match = None
    best_score = float('inf')

    for name, ref_path in reference_shapes.items():
        ref_img = cv2.imread(ref_path)
        ref_contour = get_largest_contour(ref_img, 0, 0, 0)
        if ref_contour is None:
            continue

        ref_hu = get_hu_moments(ref_contour)
        score = np.sum(np.abs(np.log10(np.abs(target_hu)) - np.log10(np.abs(ref_hu))))

        if score < best_score:
            best_score = score
            best_match = name

    return best_match


def match_shape(anomaly_hu, shape_hus):
    """
    Anomali Hu momentleri ile önceden hesaplanmış referans Hu momentlerini karşılaştırır.
    """
    scores = {}
    for name, hu in shape_hus.items():
        diff = np.sum(np.abs(-np.sign(hu) * np.log10(np.abs(hu)) -
                            -np.sign(anomaly_hu) * np.log10(np.abs(anomaly_hu))))
        scores[name] = diff
    return sorted(scores.items(), key=lambda x: x[1])


def is_shape_matched(imgAnomaly, maskPipe, YoverX, y):
    """
    Anomali maskesini yeniden şekillendirip Hu momentleri ile eşleştirir.
    """
    new = resize(imgAnomaly, maskPipe, YoverX, y)

    shape_templates = {
        "ucgen": cv2.imread("templates/ucgen.png", 0),
        "kare": cv2.imread("templates/kare.png", 0),
        "dikdortgen": cv2.imread("templates/dikdortgen.png", 0),
    }

    shape_hus = {}
    for name, template in shape_templates.items():
        _, binary = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        shape_hus[name] = cv2.HuMoments(cv2.moments(cnt)).flatten()

    contours, _ = cv2.findContours(new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        anomaly_cnt = max(contours, key=cv2.contourArea)
        anomaly_hu = cv2.HuMoments(cv2.moments(anomaly_cnt)).flatten()
    else:
        return None, []

    matches = match_shape(anomaly_hu, shape_hus)
    best_match = matches[0][0] if matches else None

    return best_match, matches


def is_object_on_pipe_context(object_mask, pipe_mask, expand_px=10, threshold=0.4, debug=False):
    """
    Nesnenin borunun üstünde olup olmadığını
    borunun üstü ve altındaki komşuluk bölgesinden kontrol eder.
    """
    object_mask_bin = (object_mask > 0).astype(np.uint8)
    pipe_mask_bin = (pipe_mask > 0).astype(np.uint8)

    y_indices, x_indices = np.where(object_mask_bin)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return False

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    y_top = max(y_min - expand_px, 0)
    y_bot = min(y_max + expand_px, object_mask.shape[0])
    x_left, x_right = x_min, x_max

    upper_context = pipe_mask_bin[y_top:y_min, x_left:x_right]
    lower_context = pipe_mask_bin[y_max:y_bot, x_left:x_right]

    upper_pipe = np.sum(upper_context)
    lower_pipe = np.sum(lower_context)

    context_total = upper_context.size + lower_context.size
    context_pipe_ratio = (upper_pipe + lower_pipe) / context_total if context_total > 0 else 0

    if debug:
        print(f"Pipe ratio in context: {context_pipe_ratio:.2f}")
        debug_img = cv2.merge([object_mask_bin * 255, pipe_mask_bin * 255, np.zeros_like(pipe_mask_bin)])
        cv2.rectangle(debug_img, (x_min, y_top), (x_max, y_bot), (255, 255, 255), 1)
        cv2.imshow("Contextual Pipe Detection", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return context_pipe_ratio >= threshold


def point_line_distance(mask, point):
    """
    Verilen bir noktadan, mask içindeki boru merkez hattına (line) dik uzaklığı döner.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        rect = cv2.minAreaRect(cnt)
        centers.append(rect[0])

    px, py = point
    centers = sorted(centers, key=lambda c: abs(c[1] - py))

    above, below = None, None
    for c in centers:
        if c[1] < py and (above is None or c[1] > above[1]):
            above = c
        elif c[1] > py and (below is None or c[1] < below[1]):
            below = c

    if above is None or below is None:
        print("Yukarı ve aşağı boru merkezi bulunamadı.")
        return None

    x1, y1 = above
    x2, y2 = below

    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    distance = abs(a * px + b * py + c) / np.sqrt(a ** 2 + b ** 2)

    return distance, (a, b, c), above, below


def is_bbox_on_pipe(bbox, pipe_mask, margin=5, debug=False):
    """
    Anomali bbox'unun boru merkez hattı üzerinde veya yakınında olup olmadığını kontrol eder.
    """
    x, y, w, h = bbox

    pipe_bin = (pipe_mask > 0).astype(np.uint8)
    pipe_skel = skeletonize(pipe_bin).astype(np.uint8) * 255

    y1, y2 = int(y), int(y + h)
    x1, x2 = int(x), int(x + w)
    bbox_region = pipe_skel[y1:y2, x1:x2]

    y1 = max(0, y1 - margin)
    y2 = min(pipe_skel.shape[0], y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(pipe_skel.shape[1], x2 + margin)

    expanded_region = pipe_skel[y1:y2, x1:x2]

    if debug:
        debug_img = cv2.cvtColor(pipe_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mask = pipe_skel > 0
        ys, xs = np.where(mask)
        debug_img[ys, xs] = (0, 0, 255)
        cv2.imshow("Debug Pipe Match", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Opsiyonel: burada expanded_region üzerinde pipe skeletonun olup olmadığını kontrol edebilirsin
    # if np.sum(expanded_region > 0) > 0:
    #    return True

    return False


def is_mostly_yellow(cropped_img, threshold=0.4):
    """
    BGR olarak verilen kırpılmış görüntünün % kaçının sarı renk olduğunu döner.
    """
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / (cropped_img.shape[0] * cropped_img.shape[1])
    return yellow_ratio > threshold


def get_pipe_short_side_centers(mask):
    """
    Borunun kısa kenarlarının merkezlerini pixel cinsinden döner.
    """
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    pt1 = (box[1] + box[2]) / 2
    pt2 = (box[3] + box[0]) / 2
    return pt1, pt2


def pixel_to_camera_ray(x, y, W, H, fov_x_deg, fov_y_deg):
    """
    Piksel koordinatlarını normalize edilmiş kamera ışınına çevirir.
    """
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)
    cx, cy = W / 2.0, H / 2.0
    fx = W / (2.0 * np.tan(fov_x / 2.0))
    fy = H / (2.0 * np.tan(fov_y / 2.0))

    xn = (x - cx) / fx
    yn = (y - cy) / fy

    ray = np.array([xn, yn, 1.0])
    ray /= np.linalg.norm(ray)
    return ray


def angle_between_camera_rays(p1, p2, W, H, fov_x_deg, fov_y_deg):
    """
    İki piksel noktası arasındaki açı (derece ve radyan) hesabı.
    """
    ray1 = pixel_to_camera_ray(p1[0], p1[1], W, H, fov_x_deg, fov_y_deg)
    ray2 = pixel_to_camera_ray(p2[0], p2[1], W, H, fov_x_deg, fov_y_deg)

    dot = np.clip(np.dot(ray1, ray2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg, angle_rad


def is_size_matched(maskPipe, tag, points, fov_y=90, focal=0, diaR=125):
    """
    Anomali boyutunun boruya göre uygunluğunu kontrol eder.
    """
    anomaly = {
        "ucgen": (296, 255),
        "yildiz": (380, 362),
        "kare": (300, 300),
        "elips": (300, 400),
        "trombus": (300, 400),
        "besgen": (353, 335),
        "dort yaprakli yonca": (310, 310),
        "altigen": (346, 300),
        "dikdortgen": (200, 400),
        "daire": (300, 300),
    }

    p1, p2 = points
    W, H = maskPipe.shape[1], maskPipe.shape[0]
    fov_x = 90

    (diameter_px2, diameter_px1), angle = findDiameters(maskPipe, (p1[1] + p2[1]) / 2)
    deg, rad = angle_between_camera_rays(p1, p2, W, H, fov_x, fov_y)
    f = focal

    d1 = diaR * f / diameter_px1
    d2 = diaR * f / diameter_px2
    BeksiA = np.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * np.cos(deg))  # Boruya paralel eksende anomalinin boyu
    BartıA = (d2 ** 2 - d1 ** 2) / BeksiA  # İşlem gereği
    b = (BeksiA + BartıA) / 2  # Uzaktaki boru çapının x ekseni değeri
    h = np.sqrt(d2 ** 2 - b ** 2)  # Yükseklik y ekseni değeri
    YoverX = h / b  # x,y=w,h anomali y ekseni normalizasyonu

    orig = anomaly[tag][0] * (1 / np.sin(angle))

    return BeksiA / orig, YoverX

def is_size_matched(maskPipe, tag, points, fov_y=90, focal=1, diaR=125):
    """
    Boru maskesi ve anomali şekli bilgisine göre anomali boyutlarını hesaplar.

    Args:
        maskPipe (np.array): Boru maskesi (binary).
        tag (str): Anomali şeklinin adı (örn. "ucgen", "kare").
        points (tuple): İki nokta koordinatı (p1, p2).
        fov_y (float): Kamera dikey görüş açısı derece cinsinden.
        focal (float): Kamera odak uzaklığı (normalizasyon için, 1 önerilir).
        diaR (float): Boru gerçek çapı (örnek: 125 mm).

    Returns:
        tuple: (anomali boy oranı, YoverX oranı)
    """

    # Anomali türlerine göre referans boyutlar (x ekseni, y ekseni) mm cinsinden
    anomaly = {
        "ucgen": (296, 255),
        "yildiz": (380, 362),
        "kare": (300, 300),
        "elips": (300, 400),
        "trombus": (300, 400),
        "besgen": (353, 335),
        "dort yaprakli yonca": (310, 310),
        "altigen": (346, 300),
        "dikdortgen": (200, 400),
        "daire": (300, 300),
    }

    p1, p2 = points
    W, H = maskPipe.shape[1], maskPipe.shape[0]
    fov_x = 90  # Kamera yatay görüş açısı varsayımı

    # Boru çaplarını piksel cinsinden ve ana açı bilgisiyle hesapla
    (diameter_px2, diameter_px1), angle = findDiameters(maskPipe, (p1[1] + p2[1]) / 2)
    
    # İki noktanın kameradaki açısını hesapla (radyan ve derece)
    deg, rad = angle_between_camera_rays(p1, p2, W, H, fov_x, fov_y)

    # Odak uzaklığı (f) ile normalize edilmiş gerçek çaplara dönüştürme
    d1 = diaR * focal / diameter_px1
    d2 = diaR * focal / diameter_px2

    # Anomali boru paralel eksenindeki uzunluğu (Law of cosines)
    BeksiA = np.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * np.cos(deg))

    # İşlem için ara değerler
    BartıA = (d2 ** 2 - d1 ** 2) / BeksiA
    b = (BeksiA + BartıA) / 2
    h = np.sqrt(d2 ** 2 - b ** 2)

    # Yükseklik / genişlik oranı
    YoverX = h / b

    # Referans uzunluk, anomaliye göre düzeltme açısı ile
    orig = anomaly[tag][0] / np.sin(angle)

    # Normalize edilmiş anomali uzunluğu
    size_ratio = BeksiA / orig

    return size_ratio, YoverX