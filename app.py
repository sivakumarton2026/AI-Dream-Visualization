import streamlit as st
import torch
import cv2
import os
import numpy as np
from diffusers import StableDiffusionPipeline
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="AI Dream Visualization",
    page_icon="✨",
    layout="centered"
)

st.title("✨ AI Dream Visualization")
st.write("Turn your dream into a cinematic AI video.")

# -----------------------------
# Configuration
# -----------------------------
KEYFRAMES = 2
FRAMES_PER_KEY = 8
FPS = 6

OUTPUT_DIR = "output"
FRAMES_DIR = "frames"

import hashlib

KEYFRAMES_DIR = "keyframes"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(KEYFRAMES_DIR, exist_ok=True)

# -----------------------------
# Load AI Model
# -----------------------------
@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe, device


# -----------------------------
# Generate Video
# -----------------------------
def generate_video(prompt, pipe, device):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
    keyframes_dir = os.path.join(KEYFRAMES_DIR, prompt_hash)
    os.makedirs(keyframes_dir, exist_ok=True)

    key_images = []

    progress = st.progress(0)
    status = st.empty()

    # -------------------------
    # Generate / Reuse Keyframes
    # -------------------------
    for i in range(KEYFRAMES):

        keyframe_path = os.path.join(
        keyframes_dir,
        f"keyframe_{i}.png"
)

        if os.path.exists(keyframe_path):

            status.write(
                f"♻️ Reusing keyframe {i + 1}/{KEYFRAMES}"
            )

            image = cv2.imread(keyframe_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        else:

            status.write(
                f"🎨 Generating keyframe {i + 1}/{KEYFRAMES}"
            )

            scene_prompt = (
                prompt
                + f", surreal dream scene, cinematic lighting, "
                  f"concept art, scene {i}"
            )

            image = pipe(
                scene_prompt,
                guidance_scale=8,
                num_inference_steps=8
            ).images[0]

            image = np.array(image)

            cv2.imwrite(
                keyframe_path,
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

        key_images.append(image)

        progress.progress((i + 1) / KEYFRAMES)

    # -------------------------
    # Create Animation Frames
    # -------------------------
    status.write("🎬 Creating animation...")

    # Remove old frames
    for file in os.listdir(FRAMES_DIR):

        file_path = os.path.join(
            FRAMES_DIR,
            file
        )

        if os.path.isfile(file_path):
            os.remove(file_path)

    count = 0

    for i in range(len(key_images) - 1):

        img1 = key_images[i]
        img2 = key_images[i + 1]

        for j in range(FRAMES_PER_KEY):

            alpha = j / FRAMES_PER_KEY

            blend = cv2.addWeighted(
                img1,
                1 - alpha,
                img2,
                alpha,
                0
            )

            h, w, _ = blend.shape

            zoom = 1 + j * 0.002

            nh = int(h / zoom)
            nw = int(w / zoom)

            y1 = (h - nh) // 2
            x1 = (w - nw) // 2

            crop = blend[
                y1:y1 + nh,
                x1:x1 + nw
            ]

            frame = cv2.resize(
                crop,
                (w, h)
            )

            frame_path = os.path.join(
                FRAMES_DIR,
                f"frame_{count:04d}.png"
            )

            cv2.imwrite(
                frame_path,
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )
            )

            count += 1

    # -------------------------
    # Build Video
    # -------------------------
    status.write("🎥 Building video...")

    files = sorted(
        [
            os.path.join(FRAMES_DIR, f)
            for f in os.listdir(FRAMES_DIR)
            if f.endswith(".png")
        ]
    )

    clip = ImageSequenceClip(
        files,
        fps=FPS
    )

    video_path = os.path.join(
        OUTPUT_DIR,
        "dream.mp4"
    )

    clip.write_videofile(
        video_path,
        fps=FPS
    )

    status.success("✅ Dream video generated!")

    return video_path


# -----------------------------
# User Interface
# -----------------------------
dream = st.text_area(
    "🌙 Describe your dream",
    placeholder=(
        "Example: A magical forest with glowing "
        "butterflies and a beautiful moon..."
    ),
    height=120
)

if st.button(
    "🎬 Generate Dream",
    use_container_width=True
):

    if not dream.strip():

        st.warning(
            "⚠️ Please describe your dream first."
        )

    else:

        with st.spinner(
            "Loading AI model..."
        ):

            pipe, device = load_model()

        st.info(
            f"Using device: {device}"
        )

        video_path = generate_video(
            dream.strip(),
            pipe,
            device
        )

        st.subheader("🎥 Your Dream Video")

        video_file = open(
            video_path,
            "rb"
        ).read()

        st.video(video_file)

        st.download_button(
            "📥 Download Dream Video",
            data=video_file,
            file_name="dream.mp4",
            mime="video/mp4",
            use_container_width=True
        )