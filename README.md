# Padel Game Analytics - Shot Classification System

## Project Overview

This project is a simple computer vision prototype for padel match analysis.

The goal of this project is to:
- detect players, rackets, and ball
- track gameplay events
- classify simple shot types
- save output in structured format


---

## Project Goal

The task is to build a basic system that can analyze padel match footage and perform shot classification.

The system should:
1. detect and track players, rackets, and ball
2. classify at least 2-3 shot types
3. save output in JSON or CSV format

---

## Features

This project includes:
- player detection using a fine-tuned YOLO model
- racket detection using a fine-tuned YOLO model
- ball detection using motion-based candidate detection
- ball tracking using temporal continuity and speed-based scoring
- shot classification using rule-based logic
- JSON and CSV output generation
- annotated output video
- simple shot count summary

---

## Project Structure

```text
Computer Vision/
├── data/
│   ├── raw/
│       ├──input_sample_video.mp4
│       ├──inference_sample_video.mp4
│   ├── frames1/
│   └── dataset/ 
│       ├──test/
       │   ├──images/
       │   └──labels/
       └──train/
           ├──images/
           └──labels/
├── runs/
├── outputs/
├── src/
│   ├── extract_frames.py
│   ├── train_detector.py
│   ├── detector.py
│   ├── tracker.py
│   ├── shot_classifier.py
│   └── infer.py
├── requirements.txt
```

---

## Dataset

The dataset consists of:
- one long video for training and annotation
- one short video for testing

There was no ready-made labeled dataset.

Because of this:
1. frames were extracted from the long video
2. the frames were manually annotated
3. the annotation classes were:
   - `player`
   - `ball`
   - `racket`
4. the annotated frames were used to fine-tune a YOLO model

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries used:
- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas

---

## How to Run

### Step 1: Extract frames from the long video

```bash
python src/extract_frames.py
```

This extracts frames from the long training video for annotation.

### Step 2: Annotate frames

Annotate the extracted frames using a tool such as:
- Roboflow  (I used this for annotation)
- CVAT
- LabelImg

Use these classes:
- `player`
- `ball`
- `racket`

Export the dataset in YOLO format.

### Step 3: Train the detector

```bash
python src/train_detector.py
```

This fine-tunes the YOLO model on the annotated dataset.

### Step 4: Run inference on the short test video

```bash
python src/infer.py
```

This runs the full pipeline on the short test video.

---

## Methodology

### 1. Player and Racket Detection

A YOLO model was fine-tuned using manually annotated frames from the training video.

This model is used to detect:
- players
- rackets

These two classes performed reasonably well after training.

### 2. Ball Detection

Ball detection was the most difficult part of the project.

The ball is:
- very small
- very fast
- sometimes blurred
- sometimes visually similar to white shoes or bright areas

The trained YOLO model was not reliable enough for ball detection, so a hybrid method was used.

The final ball detection method uses:
- frame differencing between consecutive frames
- brightness filtering
- size filtering
- circularity filtering
- rejection of candidates inside player boxes
- rejection of candidates inside racket boxes

This reduces false positives from:
- shoes
- player body parts
- racket highlights
- bright static regions

### 3. Ball Tracking

The ball detector generates several possible ball candidates.

A tracker then selects the most likely candidate using:
- previous ball position
- estimated ball speed
- distance from predicted location
- candidate confidence

This helps improve continuity across frames.

### 4. Shot Classification

Shot classification is rule-based.

The system uses:
- tracked ball trajectory
- nearest player
- direction change of the ball
- ball position relative to the player

The current shot types are:
- `forehand`
- `backhand`
- `serve_or_smash`

This is a prototype approach and not a fully trained shot classification model.

---

## Outputs

After running inference, the project generates:

### 1. Annotated Video
- `annotated_inference.mp4`

This video shows:
- detected players
- detected rackets
- detected ball
- predicted shot labels

### 2. JSON Output
- `shots.json`

This file stores structured shot predictions such as:
- frame number
- timestamp
- shot type
- confidence
- player information if available

### 3. CSV Output
- `shots.csv`

This contains the same prediction results in CSV format.

### 4. Summary Output
- `summary.json`

This stores simple shot analytics such as:
- total number of shots
- total forehands
- total backhands
- total serve or smash shots

---

## Example Output Format

Example JSON entry:

```json
{
  "frame": 210,
  "timestamp": 8.4,
  "player_id": null,
  "shot_type": "forehand",
  "confidence": 0.73
}
```

Example CSV columns:
- `frame`
- `timestamp`
- `player_id`
- `shot_type`
- `confidence`

---

## Challenges Faced

### 1. Ball Detection Was the Hardest Problem

The ball is a very small object and moves very fast.

Main problems:
- fast ball becomes blurred
- slow ball looks similar to shoes
- bright areas create false positives
- ball can overlap visually with player or racket

### 2. Limited Data

Only one long video was provided for training data creation.
There was no labeled dataset.
Only 150 frames were extracted from the video, manually annotated, and augmented using Roboflow.

### 3. GPU Memory Limitation

The available GPU had limited memory.
Because of this, training had to use smaller image sizes and smaller batch sizes.

### 4. Continuous Tracking

Even when the ball was detected in some frames, it was not always detected continuously.
This required additional tracking and filtering logic.

---

## Limitations

This is a prototype, so it still has some limitations:

- fast airborne ball is harder to detect than slow ball
- exact contact frame is approximate
- shot classification is rule-based, not learned from temporal training data
- player identity tracking is basic
- the system is mainly tuned for the provided videos and camera view

---

## Improvements and Future Work

If more time and more labeled data were available, the following improvements would be useful:

### 1. Better Ball Detector
Train a dedicated ball detector with:
- more ball-focused annotations
- more fast airborne ball examples
- more diverse court positions

### 2. Better Tracking
Use stronger tracking methods such as:
- Kalman filtering
- trajectory smoothing
- stronger re-identification logic

### 3. Court Understanding
Add court keypoint detection or court segmentation to improve:
- spatial reasoning
- bounce detection
- shot direction analysis

### 4. Better Shot Classification
Train a temporal model using labeled shot sequences for:
- forehand
- backhand
- smash
- volley
- serve

### 5. Better Analytics
Add:
- shot count per player
- rally statistics
- bounce location analysis
- shot direction analysis

### 6. Better Visualization
Improve the output video by adding:
- ball trajectory line
- contact point marker
- better overlays
- per-player statistics

---

## Key Design Decisions

Some practical decisions were made during this project:

- YOLO was used for player and racket detection
- motion-based logic was used for ball detection
- ball detections inside player and racket boxes were rejected
- speed and temporal continuity were used for ball tracking
- shot classification was implemented using rules instead of a trained sequence model

These decisions were chosen because they are practical for a prototype with limited data.

---

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas

---

## Conclusion

This project demonstrates a full computer vision pipeline for padel gameplay analysis.

It combines:
- object detection
- motion-based ball localization
- temporal tracking
- rule-based shot classification
- structured output generation

The system is not perfect, but it is a practical and meaningful prototype built under limited data conditions.

The main learning from this project is that for sports analytics, especially for very small and fast objects like a ball, combining:
- detection
- motion
- tracking
- rule-based reasoning

is often more useful than relying on only one method.

---

## Demo Video

The demo output video can be viewed here:

- Output Demo Video: [Google Drive Link](https://drive.google.com/drive/folders/1bX0WdUEeNtNh7xgCSl-LUSIyUEwBAGw1?usp=drive_link)

## Model Weights

The trained model weights can be downloaded here:

- YOLO Detector Weights (`best.pt`): [Google Drive Link](https://drive.google.com/drive/folders/11MUWoXA6QRaBqiLJh01k1LuyKFnp4RY8?usp=drive_link

```