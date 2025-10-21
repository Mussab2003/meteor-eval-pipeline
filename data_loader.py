import json


class DataLoader:
    def __init__(self, prediction_source, gt_source):
        self.prediction_source = prediction_source
        self.gt_source = gt_source

    def load_data(self):
        try:    
            print(f"Loading predictions from {self.prediction_source}")
            self.pred =  json.loads(open(self.prediction_source).read())
        except Exception as e:
            print(f"Error loading predictions: {e}")
            self.pred = None
        try:
            print(f"Loading ground truth from {self.gt_source}")
            self.gt = json.loads(open(self.gt_source).read())
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            self.gt = None

        return self.pred, self.gt

    def convert_gt_to_coco_format(self):
        for ann in self.gt['annotations']:
            x, y, w, h = ann['bbox']
            ann['bbox'] = [x, y, w, h]  # Keep COCO format [x, y, width, height]

        with open('gt_coco_format.json', 'w') as f:
            json.dump(self.gt, f)
        return self.gt


    def convert_pred_to_coco_format(self):

        if not hasattr(self, "pred") or self.pred is None:
            print("Prediction data not loaded. Run load_data() first.")
            return None

        # Track missing fields for logging
        expected_fields = {"image_id", "category_id", "bbox", "score", "object_descriptions", "logit"}
        missing_fields = expected_fields - set(self.pred[0].keys())
        if missing_fields:
            print(f"Missing fields in prediction data: {', '.join(missing_fields)}")

        coco_data = {
            "images": [],
            "annotations": []
        }

        seen_images = set()
        annotation_id = 1
        
        for pred in self.pred:
            img_id = pred.get("image_id")
            if img_id not in seen_images:
                # Create dummy image info since we don’t have file_name, height, width
                coco_data["images"].append({
                    "id": img_id,
                    "extra_info": {}
                })
                seen_images.add(img_id)

            x, y, w, h = pred.get('bbox', [])
            new_box = [x, y, w, h]  # Keep COCO format [x, y, width, height]
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "bbox": new_box,
                "extra_info" : {
                    "pred_result": {
                        "score": pred.get("score", 1),
                        "caption": pred.get("object_descriptions", "")

                    }
                }
            })

            annotation_id += 1

        with open("temp_coco_format.json", "w") as f:
            json.dump(coco_data, f)
        
        return coco_data