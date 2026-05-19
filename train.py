from ultralytics import YOLO
import os
from ultralytics.nn.modules.block import ResASPP_Optimized

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

model = YOLO(
    "")



results = model.train(
    data='',
    pretrained=False,
    epochs=400,
    imgsz=1280,
    batch=16,
    device='',
    workers=8,

    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    project='',
    name='',
    patience=30,
    save=True,
    val=True,
)

