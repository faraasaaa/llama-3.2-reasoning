import os
import requests
import streamlit as st
import time
from llama_cpp import Llama

MODEL_URL = "https://huggingface.co/erfyhersr/llama-1b-reasoning-gguf/resolve/main/unsloth.Q4_K_M.gguf"
MODEL_NAME = "unsloth.Q4_K_M.gguf"

# Download model if not exists
if not os.path.exists(MODEL_NAME):
    try:
        with st.spinner("Downloading model (this may take a while)..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_NAME, "wb") as f:
                f.write(response.content)
    except Exception as e:
        st.error(f"Model download failed: {str(e)}")
        st.stop()

@st.cache_resource
def load_model():
    return Llama(
        model_path=MODEL_NAME,
        n_ctx=2048,
        n_threads=1,
        verbose=False,
    )

model = load_model()

st.title("LLAMA 1B Reasoning Model (GGUF, CPU Streaming)")
st.markdown("Enter your prompt below. The model will stream its reasoning and final answer token by token.")

prompt = st.text_area(
    "Prompt",
    (
        "How do you solve the equation 5y - 6 = -10?"
    ),
    height=150,
)

if st.button("Generate"):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        # --- Generate Reasoning ---
        reasoning_prompt = f"User: {prompt}\n\nReasoning:"
        reasoning_output = ""
        st.subheader("Reasoning")
        reasoning_placeholder = st.empty()

        with st.spinner("Generating reasoning..."):
            for token in model(prompt=reasoning_prompt, max_tokens=1024, stream=True):
                reasoning_output += token["choices"][0]["text"]
                # Update the placeholder with the current output
                reasoning_placeholder.text(reasoning_output)
                time.sleep(0.05)
        st.success("Reasoning complete!")

        # --- Generate Answer ---
        answer_prompt = f"User: {prompt}\n\nReasoning: {reasoning_output}\n\nAnswer:"
        answer_output = ""
        st.subheader("Answer")
        answer_placeholder = st.empty()

        with st.spinner("Generating answer..."):
            for token in model(prompt=answer_prompt, max_tokens=512, stream=True):
                answer_output += token["choices"][0]["text"]
                answer_placeholder.text(answer_output)
                time.sleep(0.05)
        st.success("Answer complete!")
