import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO('/Users/Dinda Widi Puspita/intelligent-control-uts/best.pt')

# Open camera
cap = cv2.VideoCapture(1)

# Threshold confidence minimal
CONFIDENCE_THRESHOLD = 0.5

# Inisialisasi statistik
true_positives = 0
false_positives = 0
false_negatives = 0

prev_boxes = []  # Simpan bounding box frame sebelumnya

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

def calculate_iou(box1, box2):
    """
    Hitung Intersection over Union (IoU) antara dua kotak.
    box = [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area_box2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    if union == 0:
        return 0.0
    else:
        return intersection / union

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame.")
            break

        # Deteksi objek
        results = model(frame)

        # Ambil prediksi
        boxes = results[0].boxes  # boxes berisi prediksi bounding box
        current_boxes = []
        detections = 0

        for box in boxes:
            confidence = box.conf.item()
            if confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                current_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                detections += 1

        # Update metrik
        if detections > 0:
            true_positives += detections
        else:
            false_negatives += 1

        if detections > 1:
            false_positives += (detections - 1)

        total_predictions = true_positives + false_positives
        total_ground_truths = true_positives + false_negatives

        if total_ground_truths > 0 and total_predictions > 0:
            precision = true_positives / total_predictions
            recall = true_positives / total_ground_truths
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = true_positives / (total_ground_truths + false_positives)
        else:
            precision, recall, f1, accuracy = 0, 0, 0, 0

        # Hitung IoU rata-rata antar frame sebelumnya
        ious = []
        if prev_boxes and current_boxes:
            for curr_box in current_boxes:
                best_iou = 0
                for prev_box in prev_boxes:
                    iou = calculate_iou(curr_box, prev_box)
                    if iou > best_iou:
                        best_iou = iou
                ious.append(best_iou)

        avg_iou = np.mean(ious) if ious else 0.0

        # Simpan current box untuk frame berikutnya
        prev_boxes = current_boxes

        # Plot hasil
        annotated_frame = results[0].plot()

        # Tambahkan metrik ke frame
        cv2.putText(annotated_frame, f'Precision: {precision:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Recall: {recall:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'F1 Score: {f1:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Accuracy: {accuracy:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Avg IoU: {avg_iou:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Tampilkan frame
        cv2.imshow("Penambat Rel - Real-Time Detection + Metrics + IoU", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
