# Pose Extraction

## Table of Content
* Project Structure
* Requirements
* Training and testing

### Project structure

```bash
├── data
|   ├── coco
|   ├── mpii
|   ├── ...
├── experiment
├── pose
|   ├── datasets
|   ├── loss
|   ├── models
|   ├── utils
|   ├── nms
├── tools
```

### Requirements

* Ubuntu16.04/MacOS
* PyTorch/OpenCV

### Training and Testing

```bash
python scripts/train_and_evaluate.py configs/train_evaluate.yaml
```

### Inference

```bash
python scripts/estimate.py configs/inference.yaml
```