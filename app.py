import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ← error kill
from ultralytics import YOLO
import cv2

# Model load
model = YOLO('best.pt')

# Prices (apne hisaab se change kar sakte ho)
prices = {
    'Object-1': 1,
    'Object-2': 2,
    'Object-3': 3,
    'Object-4': 5,
    'Object-5': 10,
}

cap = cv2.VideoCapture("download.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================= VIDEO SAVE SETUP =================
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

output_path = "output_VID_20260307_232654.mp4"   # ← yahan naam change kar sakte ho
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
print(f"✅ Video saving ON → {output_path}")
# ===================================================

cart = {}
total = 0

print("AI Cashierless Checkout System Started! (YOLO26 + Tracking + Video Save)")
print("Press 'q' to quit | Press 'r' to reset total")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tracking
    results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.5)
    annotated_frame = frame.copy()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            
            class_name = model.names[cls_id]
            price = prices.get(class_name, 0)
           
            if track_id != -1 and track_id not in cart:
                cart[track_id] = price
                total += price
                print(f"✅ Added: {class_name} - ${price} | Total: ${total}")

            # Box + label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{class_name} ${price} (ID:{track_id}) {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Total bill on screen
    cv2.putText(annotated_frame, f"TOTAL BILL: ${total}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # === FRAME SAVE ===
    out.write(annotated_frame)
    # ================

    cv2.imshow('AI Cashierless Checkout - YOLO26 Tracking', annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        cart.clear()
        total = 0
        print("Total reset ho gaya!")

cap.release()
out.release()          # ← yeh zaroori hai warna video nahi banega
cv2.destroyAllWindows()

print(f"Session ended. Final Total: ${total}")
print(f"✅ Video saved successfully: {output_path}")