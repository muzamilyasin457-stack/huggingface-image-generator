import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os

st.set_page_config(page_title="Stick Figure Generator", layout="centered")
st.title("🖍️ Stick Figure Prompt-to-Image")

prompt = st.text_area("✏️ Describe your image prompt:", 
"""
Stick figures: Head is a perfect circle, eyes are two small black dots, body and limbs made from clean black lines (hand-drawn pencil sketch look). Background: Warm-orange textured backdrop (as provided in reference image). Style: All figures must be bold, centered, and clearly visible with high contrast against the background. A confident stick figure is shown walking into a room full of other stick figures. All other figures stop what they’re doing and silently turn to look at the entering figure. Their faces express subtle respect or curiosity. No speech or action — just quiet attention focused on the one entering.
""")

if st.button("🎨 Generate Image"):
    st.info("⏳ Loading model... please wait.")

    model_id = "stabilityai/stable-diffusion-2-1-base"  # <-- Open access
    token = os.getenv("HF_TOKEN")  # Automatically injected by Streamlit Secrets

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    with st.spinner("✨ Generating..."):
        image = pipe(prompt).images[0]
        st.image(image, caption=prompt)
