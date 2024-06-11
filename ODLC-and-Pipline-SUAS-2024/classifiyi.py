from ultralytics import YOLO
model=YOLO('last.pt')
model.predict('output2', save = True)
