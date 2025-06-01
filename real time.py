import cv2
from ultralytics import YOLO

# Load model custom (file .pt dari dataset penambat rel)
model = YOLO('/Users/HP/UTS/best (1).pt')  # Ganti dengan path lokal jika dijalankan di luar

# Buka kamera (0 = default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek di frame
    results = model(frame)

    # Plot hasil deteksi langsung di frame
    annotated_frame = results[0].plot()

    # Tampilkan hasil deteksi
    cv2.imshow("Penambat Rel - Real-Time Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
