# Screen Detector Service

This repository contains a minimal solution for detecting whether a receipt image is a screenshot (screen) vs a real photo using a MobileNetV2-based classifier and a simple Flask API.

## Project Structure
```
ml_tt/
├── data_preparation.py    # unzip, label by filename rule, split 70/15/15
├── model.py               # model architecture (MobileNetV2 transfer learning)
├── train.py               # training script
├── app.py                 # Flask API
├── requirements.txt       # dependencies
├── Dockerfile             # docker image
├── README.md              # this doc
├── models/                # saved models (output)
├── files.zip              # dataset archive (input)
└── data/                  # prepared data (output of preparation)
```

## 1) Data preparation
The preparation follows the guide:
- Unzips `files.zip` into `data/raw`.
- Labels images by filename rule:
  - Filenames containing `original` -> class `screen` (label=1)
  - Others -> class `real` (label=0)
- Splits into train/val/test = 70%/15%/15%
- Creates directory structure for Keras generators under `data/split/{train,val,test}/{real,screen}` and copies files.

Run:
```bash
python data_preparation.py
```

## 2) Training
Trains a MobileNetV2-based model with frozen base and a small head. Augmentations: horizontal flip, rotation ±15°, rescale /255.

Run:
```bash
python train.py
```
The best model is saved to `models/screen_detector.h5`.

## 3) API
Start the API locally:
```bash
python app.py
```
Predict:
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:5001/predict
```
Response:
```json
{"probability": 0.85}
```

Health check:
```bash
curl http://localhost:5001/health
```

## 4) Docker
Build and run:
```bash
docker build -t screen-detector .
docker run -p 5001:5001 screen-detector
```
Note: The Docker image expects a trained model at `models/screen_detector.h5`. You can either bake it into the image by training before building, or mount a volume with the file.

## 5) Tests
There is a simple smoke-test script for the API: `tests/tests.sh`.
It sends sample images from the `tests/` folder to the `/predict` endpoint and prints the responses.

Steps:
- Ensure the model is available at `models/screen_detector.h5` (train it first using the instructions above).
- Start the API locally:
  ```bash
  python app.py
  ```
- Make the script executable (one time):
  ```bash
  chmod +x tests/tests.sh
  ```
- Run the tests:
  ```bash
  ./tests/tests.sh
  ```

Configuration:
- By default, the script targets `http://localhost:5001/predict`.
- You can change the endpoint via an environment variable:
  ```bash
  ENDPOINT=http://127.0.0.1:5001/predict ./tests/tests.sh
  ```
- The script automatically skips missing files in `tests/` and only checks those that exist.

Docker + tests:
- Run the container:
  ```bash
  docker run -p 5001:5001 screen-detector
  ```
- Then on the host run:
  ```bash
  ./tests/tests.sh
  ```
