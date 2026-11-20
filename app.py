import streamlit as st
import torch
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
from torchvision.models.video import r3d_18, R3D_18_Weights

st.set_page_config(page_title="Human Activity Recognition", page_icon="🎥", layout="wide")
st.title("🎥 Human Activity Recognition Using Deep Learning")
st.write("Upload a short video and let a pretrained 3D CNN model identify the action.")

@st.cache_resource
def load_model():
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.eval()
    return model, weights, weights.transforms()

def read_video_frames(video_path, number_of_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Could not read this video. Try another MP4 video.")
    indices = np.linspace(0, total - 1, number_of_frames).astype(int)
    frames = []
    current = 0
    wanted = set(indices.tolist())
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        current += 1
    cap.release()
    if not frames:
        raise ValueError("No frames could be extracted from this video.")
    while len(frames) < number_of_frames:
        frames.append(frames[-1].copy())
    return frames[:number_of_frames]

uploaded = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
if uploaded:
    suffix = "." + uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name

    st.video(uploaded)
    if st.button("Analyze Activity", type="primary"):
        try:
            with st.spinner("Loading deep learning model and analyzing video..."):
                model, weights, preprocess = load_model()
                frames = read_video_frames(video_path)
                clip = torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames])
                clip = preprocess(clip)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                with torch.no_grad():
                    output = model(clip.unsqueeze(0).to(device))
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    scores, indices = torch.topk(probabilities, 5)
                categories = weights.meta["categories"]
            st.subheader("Prediction Results")
            best_label = categories[indices[0].item()]
            st.success(f"Most likely activity: **{best_label}**")
            st.write("Top 5 predicted activities:")
            for score, idx in zip(scores.cpu().tolist(), indices.cpu().tolist()):
                label = categories[idx]
                st.write(f"**{label}** — {score*100:.2f}%")
                st.progress(float(score))
            st.caption(f"Processing device: {device}. Predictions are model estimates, not guaranteed labels.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        finally:
            try:
                os.remove(video_path)
            except Exception:
                pass

st.divider()
st.subheader("About this project")
st.write("""
This application uses R3D-18, a 3D Convolutional Neural Network trained on the
Kinetics-400 action recognition dataset. Unlike a normal image CNN, a 3D CNN
learns spatial patterns and changes across video frames, helping it recognize
actions over time.
""")
