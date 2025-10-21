# Modular Evaluation Pipeline for Dense Captioning Models

A comprehensive evaluation framework for dense captioning models that provides standardized metrics and supports multiple data formats.

## 🎯 Objective

This project develops a **modular evaluation pipeline** for dense captioning models that:

- **Standardizes evaluation metrics** across different research frameworks
- **Provides GRIT score evaluation** since GRIT lacks its own evaluation module
- **Supports multiple data formats** (COCO, custom formats, mixed formats)
- **Bridges evaluation methodologies** between DenseCap 2016 and ControlCap 2024

## 🚀 Features

- ✅ **Multi-format Support**: Handles COCO format, custom formats, and mixed data structures
- ✅ **Comprehensive Metrics**: mAP, Detection mAP, IoU-based metrics, METEOR scores
- ✅ **Flexible Architecture**: Modular design for easy extension and customization
- ✅ **Robust Error Handling**: Comprehensive validation and error recovery
- ✅ **Performance Optimized**: Efficient processing of large datasets
- ✅ **Easy Integration**: Simple command-line interface and programmatic API

## 📋 Requirements

- Python 3.7+
- Java Runtime Environment (for METEOR scoring)
- Required Python packages:
  ```bash
  pip install pycocotools torch torchvision tqdm numpy
  ```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd meteor_pred_pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download METEOR JAR** (if not included):
   - Download `meteor-1.5.jar` and place it in the `meteor/` directory

## 📖 Usage

### Command Line Interface

```bash
python main.py --pred_file predictions.json --gt_file ground_truth.json
```

### Programmatic Usage

```python
from eval_densecap_grit_repo import DenseCapEvaluator

# Initialize evaluator
evaluator = DenseCapEvaluator()

# Add results
evaluator.add_result(scores, boxes, text, target_boxes, target_text, img_info)

# Compute metrics
metrics = evaluator.evaluate()
print(f"mAP: {metrics['map']:.3f}")
```

## 📊 Supported Data Formats

### Ground Truth (COCO Format)
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "caption": "a person walking",
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]]
    }
  ]
}
```

### Predictions (Flexible Format)
```json
{
  "predictions": [
    {
      "image_id": 1,
      "bbox": [x, y, width, height],
      "score": 0.95,
      "object_descriptions": "a person walking",
      "segmentation": [[x1, y1, x2, y2, ...]]
    }
  ]
}
```

## 📈 Evaluation Metrics

The pipeline computes comprehensive metrics:

- **mAP (mean Average Precision)**: Primary dense captioning metric
- **Detection mAP**: Object detection performance
- **IoU Thresholds**: 0.3, 0.4, 0.5, 0.6, 0.7
- **METEOR Thresholds**: 0, 0.05, 0.1, 0.15, 0.2, 0.25
- **Precision-Recall Curves**: Detailed performance analysis

## 🏗️ Architecture

```
Input Files → Data Formatter → Report Metrics → ControlCap Evaluator Bridge → METEOR JAR
     ↓              ↓                    ↓                        ↓              ↓
  gt.json    Format Conversion    DenseCapEvaluator         METEOR Bridge    Java Process
  pred.json     & Validation       Core Logic              Communication    METEOR Scoring
```

### Key Components

1. **`main.py`**: Entry point and orchestration
2. **`eval_densecap_grit_repo.py`**: Core evaluation logic
3. **`meteor/meteor.py`**: METEOR integration wrapper
4. **`data_loader.py`**: Data format handling and conversion

## 🔧 Configuration

### Environment Variables
- `LC_ALL=C`: Required for METEOR Java process
- Memory settings: Automatically optimized based on available RAM

### Customization
- **Special tokens**: Configure tokens to ignore during evaluation
- **IoU thresholds**: Modify overlap requirements
- **METEOR thresholds**: Adjust language quality requirements

## 📝 Example Output

```
Metrics ({
  'map': 0.164,
  'ap_breakdown': {
    'iou_0.3_meteor_0': 0.354,
    'iou_0.3_meteor_0.05': 0.315,
    ...
  },
  'detmap': 0.309,
  'det_breakdown': {
    'iou_0.3': 0.420,
    'iou_0.4': 0.371,
    ...
  }
})
```

## 🐛 Troubleshooting

### Common Issues

1. **Java not found**: Install Java Runtime Environment
2. **Memory errors**: Reduce batch size or increase available RAM
3. **Format errors**: Ensure data follows supported formats
4. **METEOR errors**: Check Java installation and memory settings

### Debug Mode
```bash
python main.py --pred_file predictions.json --gt_file ground_truth.json --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DenseCap (2016)**: Johnson et al. - Original dense captioning evaluation
- **ControlCap (2024)**: Latest controllable dense captioning work
- **METEOR**: Banerjee & Lavie - Translation evaluation metric
- **COCO**: Microsoft - Common Objects in Context dataset format

## 📚 References

- Johnson, J., et al. "DenseCap: Fully Convolutional Localization Networks for Dense Captioning." CVPR 2016.
- Banerjee, S., & Lavie, A. "METEOR: An Automatic Metric for MT Evaluation." ACL 2005.
- COCO Dataset: https://cocodataset.org/

---

**Note**: This pipeline is designed for research purposes and provides standardized evaluation metrics for dense captioning models. For production use, additional validation and testing may be required.
