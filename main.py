import argparse
from pycocotools.coco import COCO
from eval_densecap_grit_repo import DenseCapEvaluator
import tqdm
from data_loader import DataLoader
import torch
import json
import pycocotools.mask as mask_util
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description="Meteor Prediction Pipeline")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the model prediction file")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to the ground truth file")
    
    args = parser.parse_args()
    return args


def report_metrics(pred, gt):
        print(f":Begin evaluation.")

        def seg2bbox(seg):
            if isinstance(seg, list):
                seq = []
                for seg_ in seg:
                    seq.extend(seg_)
                x1, y1 = np.array(seq).reshape(-1, 2).min(0)
                x2, y2 = np.array(seq).reshape(-1, 2).max(0)
                bbox = [x1, y1, x2, y2]
            else:
                if isinstance(seg["counts"], list):
                    seg = mask_util.frPyObjects(seg, *seg["size"])
                elif not isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].encode()
                mask = mask_util.decode(seg)
                x1, x2 = np.nonzero(mask.sum(0) != 0)[0][0], np.nonzero(mask.sum(0) != 0)[0][-1]
                y1, y2 = np.nonzero(mask.sum(1) != 0)[0][0], np.nonzero(mask.sum(1) != 0)[0][-1]
                bbox = [x1, y1, x2, y2]
            return bbox

        # Load prediction data (not in COCO format)
        with open(pred, 'r') as f:
            pred_data = json.load(f)
        
        # Load ground truth data (COCO format)
        gt = COCO(gt)


        empty_pred_num = 0

        # evaluation
        ev = DenseCapEvaluator()
        recs = []
        
        # Create a mapping from image_id to predictions
        pred_by_image = {}
        if 'predictions' in pred_data:
            # Handle the case where predictions are in a 'predictions' array
            for pred in pred_data['predictions']:
                img_id = pred['image_id']
                if img_id not in pred_by_image:
                    pred_by_image[img_id] = []
                pred_by_image[img_id].append(pred)
        else:
            # Handle the case where predictions are in 'annotations' array
            for pred in pred_data['annotations']:
                img_id = pred['image_id']
                if img_id not in pred_by_image:
                    pred_by_image[img_id] = []
                pred_by_image[img_id].append(pred)
        
        for image_id, _ in tqdm.tqdm(list(gt.imgs.items())):
            anns = gt.imgToAnns[image_id]
            rec = dict()
            target_boxes = []
            target_text = []
            for ann in anns:
                box = seg2bbox(ann['segmentation'])
                target_boxes.append(box)
                target_text.append(ann['caption'])
            rec['target_boxes'] = target_boxes
            rec['target_text'] = target_text
            
            # Get predictions for this image
            preds = pred_by_image.get(image_id, [])
            if len(preds) == 0:
                empty_pred_num += 1
                continue
            scores = []
            boxes = []
            text = []
            for pred in preds:
                box = seg2bbox(pred['segmentation'])
                extra_info = pred.get('extra_info', {})
                if extra_info is None:
                    print("find empty result")
                    continue
                score = extra_info.get('score', 1)
                caption = extra_info.get('object_descriptions', "")
                scores.append(score)
                boxes.append(box)
                text.append(caption)
            rec['scores'] = scores
            rec['boxes'] = boxes
            rec['text'] = text

            rec['img_info'] = image_id
            recs.append(rec)

        for rec in tqdm.tqdm(recs):
            try:
                ev.add_result(
                    scores=torch.tensor(rec['scores']),
                    boxes=torch.tensor(rec['boxes']),
                    text=rec['text'],
                    target_boxes=torch.tensor(rec['target_boxes']),
                    target_text=rec['target_text'],
                    img_info=rec['img_info'],
                )
            except Exception as e:
                print(f"sample error: {e}")

        if empty_pred_num != 0:
            print(f":Image numbers with empty prediction ({empty_pred_num}).")

        metrics = ev.evaluate()

        print(f":Metrics ({str(metrics)}).")

        metrics["agg_metrics"] = metrics["map"]

        return metrics


def main():
    args = parse_args()
    print(f"Prediction file: {args.pred_file}")
    print(f"Ground Truth file: {args.gt_file}")

    if not args.pred_file or not args.gt_file:
        raise ValueError("Both prediction file and ground truth file must be provided.")
   


    report_metrics(args.pred_file, args.gt_file)




if __name__ == "__main__":
    main()