import os
import io
import sys
import cv2
import time
import numpy as np
import functools
from tqdm import tqdm
import torch


DEVICE = ""
IMG_SIZE = 1280
CONF_THRES = 0.001
IOU_THRES = 0.65


MODEL_PATH = ""
TEST_IMG_DIR = ""
TEST_LABEL_DIR = ""

# VisDrone2019
DATASET_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov6')))
from yolov6.utils.nms import non_max_suppression
from yolov6.data.data_augment import letterbox



_original_load = torch.load


@functools.wraps(_original_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_coco_gt_in_memory():
    print("\n" + "=" * 60)
    print("Step 1/3:")

    coco_gt_dict = {
        "images": [], "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(DATASET_CLASSES)]
    }
    img_name_to_id = {}
    ann_id = 0

    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png'))]
    for img_id, img_name in enumerate(tqdm(img_files, desc="GT")):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        img_name_to_id[img_name] = img_id
        coco_gt_dict["images"].append({"id": img_id, "file_name": img_name, "width": w, "height": h})

        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(TEST_LABEL_DIR, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        abs_w, abs_h = bw * w, bh * h
                        abs_x, abs_y = (xc * w) - (abs_w / 2), (yc * h) - (abs_h / 2)

                        coco_gt_dict["annotations"].append({
                            "id": ann_id, "image_id": img_id,
                            "category_id": cls_id + 1,
                            "bbox": [abs_x, abs_y, abs_w, abs_h], "area": abs_w * abs_h, "iscrowd": 0
                        })
                        ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    return coco_gt, img_name_to_id


def generate_predictions_and_metrics(img_name_to_id):
    print("\n" + "=" * 60)
    print(f"Step 2/3: DEVICE: GPU {DEVICE} ")

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = ckpt['model'].float().eval()
    model.to(device)

    dummy_input = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    params_m = params / 1e6
    flops_g = (macs * 2) / 1e9
    print(f"Params: {params_m:.2f} M | FLOPs: {flops_g:.2f} G (Resolution: {IMG_SIZE})")


    model.half()
    dummy_input = dummy_input.half()
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32


    for _ in range(5): model(dummy_input)

    predictions_list = []
    total_time_ms = 0.0
    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png'))]


    for img_name in tqdm(img_files, desc=f"Size: {IMG_SIZE}"):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img0 = cv2.imread(img_path)
        if img0 is None: continue


        img, ratio, (dw, dh) = letterbox(img0, new_shape=(IMG_SIZE, IMG_SIZE), stride=stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).half()
        img /= 255.0
        if len(img.shape) == 3: img = img[None]


        torch.cuda.synchronize()
        t1 = time.time()

        with torch.no_grad():
            pred = model(img)
            if isinstance(pred, (list, tuple)): pred = pred[0]
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False, max_det=300)

        torch.cuda.synchronize()
        t2 = time.time()
        total_time_ms += (t2 - t1) * 1000


        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    r_w = ratio[0] if isinstance(ratio, (tuple, list)) else ratio
                    r_h = ratio[1] if isinstance(ratio, (tuple, list)) else ratio
                    r_w, r_h = max(r_w, 1e-5), max(r_h, 1e-5)

                    x1 = (float(xyxy[0]) - dw) / r_w
                    y1 = (float(xyxy[1]) - dh) / r_h
                    x2 = (float(xyxy[2]) - dw) / r_w
                    y2 = (float(xyxy[3]) - dh) / r_h

                    predictions_list.append({
                        "image_id": img_name_to_id[img_name], "category_id": int(cls) + 1,
                        "bbox": [round(max(0.0, x1), 2), round(max(0.0, y1), 2), round(max(0.0, x2 - x1), 2),
                                 round(max(0.0, y2 - y1), 2)],
                        "score": round(float(conf), 5)
                    })

    fps = 1000.0 / (total_time_ms / len(img_files)) if total_time_ms > 0 else 0
    return predictions_list, {"params": params_m, "flops": flops_g, "fps": fps}


def evaluate_custom_ap(coco_gt, predictions_list):
    print("\n" + "=" * 60)
    print("⚡ Step 3/3:")

    coco_dt = coco_gt.loadRes(predictions_list)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.params.areaRng = [
        [0 ** 2, 1e5 ** 2], [0 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]
    ]
    coco_eval.params.areaRngLbl = ['all', 'verytiny', 'tiny', 'small', 'medium']

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_eval.evaluate();
    coco_eval.accumulate();
    coco_eval.summarize()
    sys.stdout = old_stdout

    precision = coco_eval.eval['precision']

    def get_ap(area_idx, iou_idx=None):
        try:
            p = precision[:, :, :, area_idx, -1] if iou_idx is None else precision[iou_idx, :, :, area_idx, -1]
            p = p[p > -1]
            return np.mean(p) if len(p) > 0 else 0.0
        except Exception:
            return 0.0

    return {
        "AP": get_ap(0), "AP50": get_ap(0, 0), "AP75": get_ap(0, 5),
        "AP_vt": get_ap(1), "AP_t": get_ap(2), "AP_s": get_ap(3)
    }


def main():

    coco_gt, img_name_to_id = build_coco_gt_in_memory()
    if len(coco_gt.dataset.get('annotations', [])) == 0:
        return

    pred_list, hw_metrics = generate_predictions_and_metrics(img_name_to_id)
    if not pred_list:
        return

    ap_dict = evaluate_custom_ap(coco_gt, pred_list)

    header = f"| {'Size':<6} | {'AP50':<6} | {'AP75':<6} | {'AP':<6} | {'AP_vt':<6} | {'AP_t':<6} | {'AP_s':<6} | {'Params (M)':<10} | {'FLOPs (G)':<10} | {'FPS':<6} |"
    print(header)
    print("-" * 115)

    size_str = f"{IMG_SIZE}x{IMG_SIZE}"
    row = f"| {size_str:<6} | {ap_dict['AP50']:.3f}  | {ap_dict['AP75']:.3f}  | {ap_dict['AP']:.3f}  | {ap_dict['AP_vt']:.3f}  | {ap_dict['AP_t']:.3f}  | {ap_dict['AP_s']:.3f}  | {hw_metrics['params']:.2f}       | {hw_metrics['flops']:.2f}       | {hw_metrics['fps']:.1f}   |"
    print(row)
    print("-" * 115)


if __name__ == "__main__":
    main()