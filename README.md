# Coffee-Disease-Detection

# CoffeeGuard AI

A coffee leaf disease detection system built for Rwandan smallholder farmers. It uses a convolutional neural network to classify coffee leaf images, estimate how severe the infection is, and recommend treatments.

## What it does

You upload a photo of a coffee leaf. The system tells you:

- Whether the leaf is healthy or diseased
- If diseased, what kind (rust or red spider mite)
- How bad it is (mild, moderate, or severe)
- What to do about it, with both chemical and organic options
- How much the treatment will roughly cost in Rwandan Francs

The model was trained on the RoCoLe dataset (Robusta Coffee Leaf images) with 1,560 images across 3 classes.

## Classes

| Class | Description |
|---|---|
| Healthy | No disease present |
| Coffee Leaf Rust | Fungal disease caused by *Hemileia vastatrix*. Orange-yellow spots on leaf undersides. Most common coffee disease in Rwanda. |
| Red Spider Mite | Tiny arachnids (*Oligonychus coffeae*) that cause stippling, bronzing, and webbing on leaves. |

## How the model works

The model uses MobileNetV2 pretrained on ImageNet as a feature extractor. The pretrained convolutional layers are frozen, and a custom classification head is added on top:

```
MobileNetV2 (frozen) -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(3, Softmax)
```

Training details:
- Input size: 224 x 224 x 3
- Optimizer: Adam (lr=0.001 for initial training, lr=1e-5 for fine-tuning)
- Loss: categorical crossentropy
- Data augmentation: rotation, shifts, flips, zoom, brightness changes
- Validation split: 15%
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Optional fine-tuning: last 30 layers of MobileNetV2 unfrozen with very low learning rate

## Severity estimation

After the model classifies the disease, a separate module estimates how much of the leaf is affected using color segmentation in HSV color space. It looks for brown, yellow, dark, and red pixels relative to healthy green tissue.

| Severity | Affected Area |
|---|---|
| Mild | Less than 25% |
| Moderate | 25% to 50% |
| Severe | More than 50% |

## API endpoints

The system exposes a Flask REST API:

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/` | Returns API info and available endpoints |
| POST | `/detect` | Accepts a leaf image, returns disease, severity, and treatment |
| GET | `/diseases` | Lists all detectable conditions |
| GET | `/diseases/{name}` | Returns details for one disease (healthy, red_spider_mite, rust) |

### Example: POST /detect

Send a multipart form with an image file under the key `image`.

Response:
```json
{
  "success": true,
  "detection": {
    "disease": "rust",
    "confidence": 94.2,
    "all_predictions": {
      "healthy": 2.1,
      "red_spider_mite": 3.7,
      "rust": 94.2
    }
  },
  "severity": {
    "level": "moderate",
    "percent": 35.4,
    "message": "35% affected."
  },
  "treatment": {
    "recommendation": "Moderate rust. Treat immediately.",
    "chemical": "Copper oxychloride 3g per liter every 10-14 days.",
    "organic": "Bordeaux mixture 2% with pruning.",
    "instructions": "Prune leaves over 50% covered. Burn material. Spray remaining.",
    "cost_rwf": 8500
  },
  "disease_info": {
    "name": "Coffee Leaf Rust",
    "scientific_name": "Hemileia vastatrix",
    "description": "Most damaging coffee disease in Rwanda caused by airborne fungal spores.",
    "symptoms": "Orange-yellow spots on leaf undersides, pale patches on top, leaf drop.",
    "prevention": "Resistant varieties, proper shade, pruning, copper fungicides before rainy season."
  }
}
```

## Project structure

```
coffee-disease-detector/
    app/
        __init__.py
        app.py              # Flask routes and API logic
        model.py            # MobileNetV2 model loading and prediction
        severity.py         # Color-based severity estimation
        database.py         # SQLite database for diagnostics history
        templates/
            base.html
            index.html
            about.html
            history.html
            404.html
        static/
            css/
                style.css
            js/
                detect.js
            images/
            uploads/
    models/
        coffee_disease_model.h5
    data/
    train_model.py
    run.py
    requirements.txt
    README.md
    .gitignore
```

## Setup

You need Python 3.8 or higher.

```bash
git clone https://github.com/YOUR_USERNAME/coffee-disease-detector.git
cd coffee-disease-detector
pip install -r requirements.txt
python run.py
```

Then open http://localhost:5000 in your browser.

## Requirements

- tensorflow
- flask
- opencv-python-headless
- pillow
- numpy
- scikit-learn

All listed in `requirements.txt`.

## Dataset

RoCoLe - Robusta Coffee Leaf Images Dataset from Kaggle.

- Source: https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset
- 1,560 images total
- 3 classes: healthy (791), rust (602), red_spider_mite (167)

The class imbalance (red spider mite has fewer samples) is partially addressed through data augmentation during training.

## Limitations

- The model was trained on 1,560 images, which is relatively small. Performance may vary on images that look very different from the training data.
- Severity estimation uses color thresholds that were tuned for this dataset. Unusual lighting or backgrounds may give inaccurate readings.
- The system only detects 3 conditions. Other coffee diseases (like Coffee Berry Disease or Leaf Miner) are not covered.
- The model works best on close-up photos of individual leaves with reasonable lighting.

## Context

This project was built as a capstone for the ALU Software Engineering program. Coffee is Rwanda's most important export crop, and smallholder farmers often lack access to agricultural extension services. The goal is to provide a simple tool that helps farmers identify diseases early and take action before significant yield loss occurs.

## License

This project is for educational purposes.
