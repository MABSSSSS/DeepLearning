# Cats vs Dogs Detection using YOLOv9 and AlexNet

This project demonstrates the use of two different deep learning techniques to solve the Cats vs Dogs classification and detection problem:

1. **AlexNet** - for image classification
2. **YOLOv9** - for object detection

## Dataset
Dataset used: [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/overview)

We used a manually annotated version of the dataset with YOLO-format labels. The dataset is split into:
- Training images and labels
- Validation set (optional)
- Test set

Classes:
- `0` = Cat
- `1` = Dog

## Project Structure
```
Cats-vs-Dogs-Detection-YOLOv9-AlexNet/
│
├── README.md
├── requirements.txt
├── yolov9_config.yaml
├── alexnet_model.py
├── train_alexnet.py
├── train_yolov9.sh
├── results/
│   ├── alexnet_confusion_matrix.png
│   ├── yolov9_map_curve.png
│   └── training_loss_curve.png
├── dataset/
│   ├── images/          # Images in YOLO format structure
│   ├── labels/          # .txt YOLO annotations
│   └── classes.txt      # Contains: cat\ndog
└── utils/
    └── preprocess.py    # Preprocessing utilities
```

## Setup
```bash
pip install -r requirements.txt
```

## Training AlexNet
```bash
python train_alexnet.py
```

## Training YOLOv9
```bash
bash train_yolov9.sh
```

## Results
Output graphs and performance metrics are stored in the `results/` folder.

---

📂 GitHub: [MABSSSSS/DeepLearning](https://github.com/MABSSSSS/DeepLearning)
