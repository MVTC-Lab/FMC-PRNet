from ultralytics import YOLO
import os

if __name__ == '__main__':
    model_path = ""  # best.py
    model = YOLO(model_path)

    metrics = model.val(
        data='',
        imgsz=1280,
        batch=16,
        device='0',
        workers=8,

        project='./result',
        name='FMC_PRNet_val',

    )
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@75:    {metrics.box.map75:.4f}")
