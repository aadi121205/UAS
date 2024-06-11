from ultralytics import YOLO
model=YOLO('best.pt')
model.predict('from phone', save = True, save_txt = True, save_crop = True)
