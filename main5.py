import cv2
import mediapipe as mp
import numpy as np
import time
import imutils

# --- Inisialisasi ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Landmark Mulut yang Lebih Akurat ---
MOUTH_INNER_UPPER = 13
MOUTH_INNER_LOWER = 14
MOUTH_INNER_LEFT = 78
MOUTH_INNER_RIGHT = 308
MOUTH_OUTER_LEFT = 61
MOUTH_OUTER_RIGHT = 291
MOUTH_OUTER_UPPER = 0
MOUTH_OUTER_LOWER = 17

# --- Landmark untuk senyum ---
SMILE_LEFT_CORNER = 61
SMILE_RIGHT_CORNER = 291
SMILE_UPPER_LIP = 0
SMILE_LOWER_LIP = 17

# --- Fungsi Jarak ---
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- Eye Aspect Ratio ---
def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- Deteksi Senyum (Pengecualian) ---
def is_smiling(landmarks, h, w):
    try:
        left_corner = (int(landmarks[SMILE_LEFT_CORNER].x * w), int(landmarks[SMILE_LEFT_CORNER].y * h))
        right_corner = (int(landmarks[SMILE_RIGHT_CORNER].x * w), int(landmarks[SMILE_RIGHT_CORNER].y * h))
        upper_lip = (int(landmarks[SMILE_UPPER_LIP].x * w), int(landmarks[SMILE_UPPER_LIP].y * h))
        lower_lip = (int(landmarks[SMILE_LOWER_LIP].x * w), int(landmarks[SMILE_LOWER_LIP].y * h))
        
        # Lebar senyum
        smile_width = euclidean_dist(left_corner, right_corner)
        
        # Tinggi bukaan mulut (harus kecil untuk senyum)
        smile_height = euclidean_dist(upper_lip, lower_lip)
        
        # Rasio lebar/tinggi untuk senyum
        if smile_height == 0:
            return False
            
        smile_ratio = smile_width / smile_height
        
        # Senyum biasanya memiliki lebar yang signifikan tapi tinggi kecil
        is_wide_smile = smile_width > 80  # Lebar minimal
        is_small_height = smile_height < 25  # Tinggi maksimal
        is_smile_ratio_high = smile_ratio > 3.5  # Rasio lebar/tinggi tinggi
        
        return is_wide_smile and is_small_height and is_smile_ratio_high
        
    except:
        return False

# --- DETEKSI MENGUAP YANG LEBIH AKURAT ---
def is_yawning(landmarks, h, w):
    try:
        # Titik utama bagian dalam mulut (lebih sensitif)
        inner_top = (int(landmarks[MOUTH_INNER_UPPER].x * w), int(landmarks[MOUTH_INNER_UPPER].y * h))
        inner_bottom = (int(landmarks[MOUTH_INNER_LOWER].x * w), int(landmarks[MOUTH_INNER_LOWER].y * h))
        inner_left = (int(landmarks[MOUTH_INNER_LEFT].x * w), int(landmarks[MOUTH_INNER_LEFT].y * h))
        inner_right = (int(landmarks[MOUTH_INNER_RIGHT].x * w), int(landmarks[MOUTH_INNER_RIGHT].y * h))
        
        # Titik bagian luar untuk konfirmasi
        outer_top = (int(landmarks[MOUTH_OUTER_UPPER].x * w), int(landmarks[MOUTH_OUTER_UPPER].y * h))
        outer_bottom = (int(landmarks[MOUTH_OUTER_LOWER].x * w), int(landmarks[MOUTH_OUTER_LOWER].y * h))

        # Hitung jarak vertikal dan horizontal
        inner_height = euclidean_dist(inner_top, inner_bottom)
        inner_width = euclidean_dist(inner_left, inner_right)
        
        outer_height = euclidean_dist(outer_top, outer_bottom)
        
        if inner_width == 0:
            return 0.0, False

        # MAR (Mouth Aspect Ratio) - menggunakan bagian dalam
        mar = inner_height / inner_width
        
        # Rasio tambahan untuk konfirmasi
        height_ratio = inner_height / outer_height if outer_height > 0 else 0
        
        # Kondisi menguap yang lebih akurat:
        is_mar_high = mar > 0.6
        is_height_significant = height_ratio > 0.5
        is_not_too_wide = inner_width < 150
        
        # Pastikan ini bukan senyum
        is_smile = is_smiling(landmarks, h, w)
        
        is_yawn = is_mar_high and is_height_significant and is_not_too_wide and not is_smile
        
        return mar, is_yawn

    except Exception as e:
        return 0.0, False

# --- Deteksi Seatbelt ---
def detect_seatbelt(landmarks, img_shape):
    h, w = img_shape[:2]
    try:
        left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
        x1, y1 = left_shoulder
        roi = frame[max(0, y1-50):min(h, y1+100), max(0, x1-80):min(w, x1+80)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
        coverage = cv2.countNonZero(mask) / mask.size
        return coverage > 0.15
    except:
        return False

# --- Threshold ---
EAR_THRESHOLD = 0.25
EAR_SMILE_THRESHOLD = 0.20  # Threshold lebih rendah untuk senyum
EYE_CLOSED_SHORT = 3
EYE_CLOSED_LONG = 30
YAWN_DURATION = 2

# --- Timer ---
eye_closed_start = None
yawn_start = None
alert_short = alert_long = alert_yawn = alert_seatbelt = False

# --- Variabel untuk smoothing deteksi ---
yawn_counter = 0
yawn_frames_required = 5
smile_counter = 0
smile_frames_required = 3

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Reset
    alert_short = alert_long = alert_yawn = alert_seatbelt = False
    current_yawn_detected = False
    current_smile_detected = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = face_landmarks.landmark

            # --- EAR ---
            left_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [33, 160, 158, 133, 153, 144]]
            right_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [362, 385, 387, 263, 373, 380]]
            ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0

            # --- DETEKSI SENYUM ---
            current_smile_detected = is_smiling(landmarks, h, w)
            
            # Smoothing deteksi senyum
            if current_smile_detected:
                smile_counter += 1
            else:
                smile_counter = max(0, smile_counter - 1)
            
            smoothed_smile_detected = smile_counter >= smile_frames_required

            # --- MENGUAP AKURAT ---
            mar, is_yawn_detected = is_yawning(landmarks, h, w)
            current_yawn_detected = is_yawn_detected

            # Smoothing deteksi menguap
            if is_yawn_detected:
                yawn_counter += 1
            else:
                yawn_counter = max(0, yawn_counter - 1)
            
            smoothed_yawn_detected = yawn_counter >= yawn_frames_required

            # --- Seatbelt ---
            has_seatbelt = detect_seatbelt(landmarks, frame.shape)
            if not has_seatbelt:
                alert_seatbelt = True

            # --- LOGIKA PENGEcUALIAN SENYUM YANG CERDAS ---
            is_smiling_while_eyes_closed = smoothed_smile_detected and (ear < EAR_THRESHOLD)
            
            # Threshold yang berbeda untuk senyum vs ngantuk
            if smoothed_smile_detected:
                # Saat senyum, gunakan threshold yang lebih rendah untuk mata
                eye_threshold = EAR_SMILE_THRESHOLD
                eye_status = "SENYUM - MATA NORMAL" if ear >= eye_threshold else "SENYUM - TIDAK NGANTUK"
            else:
                # Saat tidak senyum, gunakan threshold normal
                eye_threshold = EAR_THRESHOLD
                eye_status = "MATA NORMAL" if ear >= eye_threshold else "NGANTUK"

            # --- MATA TERTUTUP (dengan pengecualian senyum) ---
            if ear < eye_threshold:
                # Jika mata tertutup karena senyum, jangan hitung sebagai ngantuk
                if is_smiling_while_eyes_closed:
                    # Reset timer karena ini adalah senyum, bukan ngantuk
                    eye_closed_start = None
                    eye_status = "SENYUM (BUKAN NGANTUK)"
                else:
                    # Ini benar-benar ngantuk
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    else:
                        elapsed = time.time() - eye_closed_start
                        if elapsed >= EYE_CLOSED_LONG:
                            alert_long = True
                        elif elapsed >= EYE_CLOSED_SHORT:
                            alert_short = True
            else:
                eye_closed_start = None

            # --- MENGUAP (dengan smoothing) ---
            if smoothed_yawn_detected:
                if yawn_start is None:
                    yawn_start = time.time()
                else:
                    elapsed = time.time() - yawn_start
                    if elapsed >= YAWN_DURATION:
                        alert_yawn = True
            else:
                yawn_start = None

            # --- Gambar titik untuk visualisasi ---
            mouth_points = [MOUTH_INNER_UPPER, MOUTH_INNER_LOWER, MOUTH_INNER_LEFT, MOUTH_INNER_RIGHT,
                          SMILE_LEFT_CORNER, SMILE_RIGHT_CORNER, SMILE_UPPER_LIP, SMILE_LOWER_LIP]
            
            colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (255, 0, 0),  # Inner points
                     (255, 255, 0), (255, 255, 0), (0, 255, 255), (0, 255, 255)]  # Smile points
            
            for idx, color in zip(mouth_points, colors):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, color, -1)

            # --- Status dengan informasi lebih detail ---
            status = "NORMAL"
            color = (0, 255, 0)
            
            if alert_long: 
                status = "NGANTUK PARAH"
                color = (0, 0, 255)
            elif alert_short: 
                status = "NGANTUK"
                color = (0, 165, 255)
            elif alert_yawn: 
                status = "MENGUAP"
                color = (0, 255, 255)
            elif alert_seatbelt: 
                status = "TANPA SEATBELT"
                color = (255, 0, 0)
            elif smoothed_smile_detected:
                status = "SENYUM"
                color = (255, 0, 255)
            elif current_yawn_detected:
                status = "MENGUAP"
                color = (255, 255, 0)

            # Display information
            cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Eye: {eye_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Smile: {smile_counter}/{smile_frames_required}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Yawn: {yawn_counter}/{yawn_frames_required}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Gambar indikator visual
            if smoothed_smile_detected:
                left_corner = (int(landmarks[SMILE_LEFT_CORNER].x * w), int(landmarks[SMILE_LEFT_CORNER].y * h))
                right_corner = (int(landmarks[SMILE_RIGHT_CORNER].x * w), int(landmarks[SMILE_RIGHT_CORNER].y * h))
                cv2.line(frame, left_corner, right_corner, (255, 0, 255), 2)
                cv2.putText(frame, "SENYUM", (left_corner[0], left_corner[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    else:
        eye_closed_start = yawn_start = None
        yawn_counter = 0
        smile_counter = 0

    # --- ALERT VISUAL ---
    if alert_long:
        cv2.putText(frame, "MATA TERTUTUP LAMA!", (30, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

    if alert_short and not smoothed_smile_detected:  # Jangan tampilkan alert jika senyum
        cv2.putText(frame, "NGANTUK!", (80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3)

    if alert_yawn:
        cv2.putText(frame, "MENGUAP!", (frame.shape[1]//2 - 100, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

    if alert_seatbelt:
        cv2.putText(frame, "PAKAI SEATBELT!", (30, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    # --- Tampilkan ---
    cv2.imshow("DETEKSI NGANTUK", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()