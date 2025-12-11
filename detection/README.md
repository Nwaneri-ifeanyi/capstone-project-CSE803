# CSE803 Facial Landmark Detection Project

This folder contains a customised PyTorch implementation of **YOLOv4** that we use for the CSE803 course project. It extends the original Tianxiaomo codebase with data loaders for our annotations, evaluation hooks that log every 5 epochs.

---

## Repository Layout

| Path | Description |
| ---- | ----------- |
| `train.py` | Main training script with periodic COCO evaluation and loss/metric logging. |
| `dataset.py` | Custom dataset/augmentation pipeline that reads `label_*.txt` annotations. |
| `cfg.py` & `cfg/` | Hyper-parameter definitions and legacy Darknet configs. |
| `models.py` | YOLOv4 backbone, neck and head definitions. |
| `tool/` | Utility modules (losses, COCO adapters, Darknet converters, etc.). |
| `label_train.txt`, `label_val.txt` | Annotation manifests in `image_path x1,y1,x2,y2,class ...` format. |


Pre-converted weights (`yolov4.conv.137.pth`) are shipped locally to bootstrap training.
Download the weights from the original repository. 

---


## Training

1. **Configure hyper-parameters** in `cfg.py` (batch size, subdivisions, epochs, mixup/mosaic toggles, etc.). The script also exposes CLI overrides:

   ```bash
   python train.py \
     -pretrained yolov4.conv.137.pth \
     -classes 4
   ```

---

## Credits

This project builds on Tianxiaomo’s [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) implementation and Darknet conversion utilities by AlexeyAB. The CSE803 team adapted it for a 4-class face/object detection dataset, integrated COCO evaluation, and added convenience scripts for plotting and experimentation.

For questions or reproducibility details, reach out to palsoum1@msu.edu.
