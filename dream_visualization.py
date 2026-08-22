import torch
import cv2
from diffusers import StableDiffusionPipeline
import os
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

PROMPT = input("Describe your dream: ")

# ---- CONFIG ----
KEYFRAMES = 6
FRAMES_PER_KEY = 15
FPS = 6

OUTPUT_DIR = "output"
FRAMES_DIR = "frames"

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")

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

print(f"Using device: {device}")

print("Generating keyframes...")

key_images = []

for i in range(KEYFRAMES):

    prompt = (
        PROMPT
        + f", surreal dream scene, cinematic lighting, "
          f"concept art, scene {i}"
    )

    image = pipe(
        prompt,
        guidance_scale=8,
        num_inference_steps=25
    ).images[0]

    key_images.append(np.array(image))

    print(f"Generated keyframe {i + 1}/{KEYFRAMES}")


print("Animating frames...")

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

        frame = cv2.resize(crop, (w, h))

        frame_path = os.path.join(
            FRAMES_DIR,
            f"frame_{count:04d}.png"
        )

        cv2.imwrite(
            frame_path,
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )

        count += 1


print("Building video...")

files = sorted(
    [
        os.path.join(FRAMES_DIR, f)
        for f in os.listdir(FRAMES_DIR)
    ]
)

clip = ImageSequenceClip(files, fps=FPS)

clip.write_videofile(
    os.path.join(OUTPUT_DIR, "dream.mp4")
)

print("Done! Check output/dream.mp4")