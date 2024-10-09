from ultralytics import YOLOv10
# Load a pretrained YOLOv10n model
model = YOLOv10("./runs/detect/train11/weights/best.pt")
# Perform object detection on an image
# results = model("test1.jpg")
results = model.predict(r"D:\suxici\DATA\prod_line\paper-data\yolov5\test.txt")
# Display the results
# for item in results:
#     item.show()
results[0].show()