from ultralytics import YOLO

model = YOLO("yolo11n.pt")  #smallest model, downloads automatically on first run

results = model("cat.jpg")  #path to your photo

# Save the annotated image (box + label + confidence)
results[0].save(filename="cat_detected.jpg")

# Print the raw detections
for box in results[0].boxes:
    cls = model.names[int(box.cls)]
    conf = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{cls}: {conf:.2f} at ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")