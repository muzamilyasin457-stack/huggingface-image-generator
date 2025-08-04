# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Hugging Face Token (safe for testing, don’t share in public apps)
HF_TOKEN = "token here"

st.set_page_config(page_title="HuggingFace Image Generator", layout="centered")
st.title("🎨 Free Prompt-to-Image Generator (HuggingFace)")

prompt = st.text_input("📝 Enter your prompt:", placeholder="A futuristic city skyline at sunset")

if st.button("Generate Image"):
    with st.spinner("Generating... please wait"):
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        image = pipe(prompt).images[0]
        st.image(image, caption=prompt)
