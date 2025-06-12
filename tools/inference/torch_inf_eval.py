"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torchvision.transforms as T

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torchvision.ops import box_iou

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from timer import timed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def draw(images, labels, boxes, scores, file_name, thrh):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue', )

        im.save(f'../eval_results/bboxed_images/{file_name}')

@timed()
def inference(model, im_data, orig_size):
    return model(im_data, orig_size)

def process_image(model, device, file_path, iou_thr, img_size, fp16_flag):
    file_name = os.path.basename(file_path)
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    with torch.inference_mode():
        orig_size = torch.tensor([[w, h]]).to(device)

        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(device)
        with autocast(dtype=torch.float16 if fp16_flag else torch.float32):
            output, inference_time = inference(model, im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores, file_name, iou_thr)
    return output, inference_time


def precision_recall(df, predicts, iou_th, class_idx=0):
    tp = 0
    fp = 0
    fn = 0
    category_to_num = {
        "C beveling": 0,
        "Dimensional tolerance": 1,
        "Fit tolerance": 2,
        "R beveling": 3,
        "Special instructions": 4,
        "Specified tolerance": 5,
    }
    num_to_category = {value: key for key, value in category_to_num.items()}

    all_eval_results = list()

    for idx in range(len(predicts)):
        predict = predicts[idx]

        if len(predict) == 0:
            continue

        id = df.id[idx]
        image_path = df.image_path[idx]
        source_name = df.source_name[idx]

        labels, boxes, scores = predict  # unpack the tuple
        # Fix shapes
        boxes = boxes.squeeze(0)
        labels = labels.squeeze(0)
        scores = scores.squeeze(0)
        # Limit predicted values to > IOU
        mask = scores > iou_th
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        pred_bboxes = boxes

        pred_cls = labels
        pred_index = torch.where(pred_cls == class_idx)[0]
        gt = np.array(eval(df.label[idx]))
        gt_bboxes = gt[:, :4].astype(int)
        gt_bboxes = torch.Tensor(gt_bboxes)
        gt_cls = torch.Tensor([category_to_num[i] for i in gt[:, 5]])
        gt_index = torch.where(gt_cls == class_idx)[0]

        if len(pred_index) == 0:
            fn += len(gt_index)
            g_bbx = gt_bboxes[gt_index].tolist()
            all_eval_results.append(
                {
                    "id": id,
                    "image_path": image_path,
                    "source_name": source_name,
                    "class": num_to_category[class_idx],
                    "num_gt": len(gt_index),
                    "num_pred": len(pred_index),
                    "tp": 0,
                    "fp": 0,
                    "fn": len(gt_index),
                    "gt_bboxes": g_bbx,
                    "pred_bboxes": [],
                }
            )
            continue
        elif len(gt_index) == 0:
            fp += len(pred_index)
            p_bbx = pred_bboxes[pred_index]
            all_eval_results.append(
                {
                    "id": id,
                    "image_path": image_path,
                    "source_name": source_name,
                    "class": num_to_category[class_idx],
                    "num_gt": len(gt_index),
                    "num_pred": len(pred_index),
                    "tp": 0,
                    "fp": len(pred_index),
                    "fn": 0,
                    "gt_bboxes": [],
                    "pred_bboxes": p_bbx.tolist(),
                }
            )
            continue

        pred_bboxes = pred_bboxes[pred_index]
        gt_bboxes = gt_bboxes[gt_index]

        iou_matrix = box_iou(gt_bboxes, pred_bboxes)
        tmp_tp = len(torch.where(iou_matrix.max(0)[0] > iou_th)[0])
        tmp_fp = len(pred_bboxes) - tmp_tp
        tmp_fn = len(gt_bboxes) - tmp_tp

        tp += tmp_tp
        fp += tmp_fp
        fn += tmp_fn
        all_eval_results.append(
            {
                "id": id,
                "image_path": image_path,
                "source_name": source_name,
                "class": num_to_category[class_idx],
                "num_gt": len(gt_index),
                "num_pred": len(pred_index),
                "tp": tmp_tp,
                "fp": tmp_fp,
                "fn": tmp_fn,
                "gt_bboxes": gt_bboxes.tolist(),
                "pred_bboxes": pred_bboxes.tolist(),
            }
        )

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    print(f"class: {num_to_category[class_idx]} iou_th: {iou_th}")
    print(f"precision: {precision} recall: {recall}")
    print(f"tp: {tp} fp: {fp} fn: {fn}")
    return pd.DataFrame(all_eval_results)


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Loop over eval data
    predicts = []
    inference_times = []
    eval_csv = args.input
    iou_th = args.threshold
    output_csv = args.output
    fp16_flag = args.fp16
    img_size = args.size
    df = pd.read_csv(eval_csv)
    val_df = df.loc[df.cv == "val"].reset_index(drop=True)
    image_files = val_df["image_path"].tolist()
    print(f"using {iou_th} as bbox evaluation IOU threshold")

    if not image_files:
        print("No PNG images found in directory.")
    else:
        for img_path in image_files:
            print(f"Processing {img_path} ...")
            outputs, inference_time = process_image(model, device, str(img_path),iou_th, img_size, fp16_flag)
            if outputs is None:
                predicts.append([])
            else:
                labels, boxes, scores = outputs
                inference_times.append(inference_time)
                # Move to CPU and detach to save memory
                labels_cpu = labels.detach().cpu()
                boxes_cpu = boxes.detach().cpu()
                scores_cpu = scores.detach().cpu()
                predicts.append((labels_cpu, boxes_cpu, scores_cpu))
                del labels, boxes, scores
                torch.cuda.empty_cache()
            torch.save(predicts, "./deim_predicts_00.pt")
        print("Image batch processing complete.")
    # evaluation
    all_eval_dfs = list()
    for class_idx in range(6):
        eval_df = precision_recall(val_df, predicts, iou_th, class_idx=class_idx)
        all_eval_dfs.append(eval_df)
    predict_result = pd.concat(all_eval_dfs)
    predict_result.to_csv(output_csv, index=False)
    inference_times_df = pd.DataFrame(inference_times[1:], columns=['time']) # [1:] to avoid first run outlier
    inference_times_df.to_csv('./inference_times.csv', index=False)
    average = inference_times_df['time'].mean()
    minimum = inference_times_df['time'].min()
    maximum = inference_times_df['time'].max()
    print(f"Inference time average: {average:.4f} s")
    print(f"Inference time Min:     {minimum:.4f} s")
    print(f"Inference time Max:     {maximum:.4f} s")    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='DEIM config file that was used in training')
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True, help='CSV file for evaluation. Ex.:2048.csv')
    parser.add_argument('-o', '--output', type=str, default='predict_validation_data.csv', help='Output filename for evaluation results csv')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='bbox IOU threshold for evaluation')
    parser.add_argument('-s', '--size', type=int, default=2048, help='Image sizes: 2048 or 4096')
    parser.add_argument('--fp16', action='store_true', default=False, help='fp16 precision or not',)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
