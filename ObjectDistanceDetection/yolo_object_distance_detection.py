from ultralytics import YOLO
import cv2


KNOWN_OBJECTS = {
    "person": 45,       
    "bottle": 7,        
    "chair": 40,        
    "stop sign": 75,    
}

CALIBRATION_DISTANCE_CM = 70   


model = YOLO("yolov8n.pt")  

class_names = model.names
print("Loaded YOLOv8n with classes:", class_names)

CLASS_WIDTHS = {}
for cid, name in class_names.items():
    if name in KNOWN_OBJECTS:
        CLASS_WIDTHS[cid] = KNOWN_OBJECTS[name]

print("Classes with distance enabled:")
for cid, width in CLASS_WIDTHS.items():
    print(f" - {class_names[cid]} (id {cid}), width {width} cm")

if not CLASS_WIDTHS:
    print("⚠ No matching classes in KNOWN_OBJECTS. Edit KNOWN_OBJECTS dict.")
    
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

cap.set(3, 1280)
cap.set(4, 720)


focal_lengths = {cid: None for cid in CLASS_WIDTHS.keys()}

print("\nInstructions:")
print(f"- Put a known object (e.g. person at {CALIBRATION_DISTANCE_CM} cm)")
print("- Press 'c' to CALIBRATE for whatever class is in front")
print("- After calibration, distance will be shown for that class")
print("- Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame")
        break

    key = cv2.waitKey(1) & 0xFF

 
    results = model(frame)

    annotated = frame.copy()

    r = results[0]
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pixel_width = x2 - x1

        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)

        label = f"{cls_name} {conf:.2f}"

        distance_text = ""
        if cls_id in CLASS_WIDTHS and pixel_width > 0:
            known_width_cm = CLASS_WIDTHS[cls_id]
            focal = focal_lengths[cls_id]

            if key == ord('c'):
                focal = (pixel_width * CALIBRATION_DISTANCE_CM) / known_width_cm
                focal_lengths[cls_id] = focal
                print(f"✅ Calibrated '{cls_name}' (id {cls_id}): focal = {focal:.2f}")

            if focal is not None:
                distance_cm = (known_width_cm * focal) / pixel_width
                distance_text = f"{distance_cm:.1f} cm"
                label += f" | {distance_text}"

    
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        annotated,
        "Press 'c' to calibrate, 'q' to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Object + Distance Detection", annotated)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
