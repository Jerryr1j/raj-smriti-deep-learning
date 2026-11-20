# Human Activity Recognition Using Deep Learning

A ready-to-run video action recognition demo built with Python, Streamlit, PyTorch and a pretrained R3D-18 3D CNN.

## Features
- Upload MP4, AVI or MOV videos
- Extracts 16 frames from the video
- Predicts the top 5 likely action classes
- Displays confidence scores
- Uses GPU automatically when CUDA is available; otherwise uses CPU

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection on first run (to download pretrained model weights)
- 4 GB RAM minimum recommended; more is better

## Run in VS Code

### 1. Extract the ZIP and open the folder in VS Code

### 2. Create a virtual environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, use the official selector at https://pytorch.org/get-started/locally/ to select the correct command for your operating system and CPU/GPU.

### 4. Start the app

```bash
streamlit run app.py
```

The browser will open. Upload a short video and click **Analyze Activity**.

## Project structure

```text
human_activity_recognition_dl/
├── app.py
├── requirements.txt
└── README.md
```

## How it works

1. Video upload: Streamlit accepts a video file.
2. Frame sampling: OpenCV reads the video and samples 16 frames.
3. Preprocessing: Frames are resized and normalized using the model's official transforms.
4. Feature learning: R3D-18 applies 3D convolutions across height, width and time.
5. Classification: The model produces scores for 400 Kinetics action categories.
6. Output: The app shows the five highest-scoring classes.

## Important limitations

- This is an inference/demo project using pretrained weights; it does not train a model on a custom dataset.
- Kinetics-400 recognizes a broad range of actions, but it may not recognize every gym exercise or unusual activity.
- Results depend on video quality, duration, camera angle and visible action.
- Confidence scores are model probabilities, not proof that a prediction is correct.

## Viva explanation

**Why 3D CNN?** A 2D CNN mainly learns spatial features from individual images. A 3D CNN also learns temporal patterns across consecutive frames, which helps distinguish actions involving movement.

**Why R3D-18?** It is a residual 3D convolutional network designed for video classification.

**What is transfer learning?** Using a model already trained on a large dataset and applying it to a new prediction task. Here, pretrained Kinetics-400 weights are used directly.

**Dataset:** Kinetics-400, a video action recognition dataset. This demo uses the pretrained model and does not require downloading the full dataset.
