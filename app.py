import streamlit as st
import os
from PIL import Image
import base64
import io
import requests

from utils import load_and_store_documents, medical_chatbot_with_accuracy

st.set_page_config(page_title="MediQuery", layout="wide")
st.title("🩺 MediQuery – Medical Health Assistant")

pdf_folder_path = "encyclopedia_pdf"

# ---------------- LOAD PDF ONLY ONCE ----------------
@st.cache_resource
def load_docs_once():
    load_and_store_documents(pdf_folder_path)

with st.spinner("Loading medical encyclopedia..."):
    load_docs_once()

# ---------------- MODE ----------------
mode = st.radio(
    "Choose Mode:",
    ["Ask from Medical Encyclopedia", "Upload Symptom Image"]
)

# ---------------- TEXT MODE ----------------
if mode == "Ask from Medical Encyclopedia":
    st.subheader("Ask from Medical Encyclopedia")

    question = st.text_input("Enter medical question")
    reference = st.text_area("Paste reference answer")

    if st.button("Get Answer"):
        if not question or not reference:
            st.warning("Please enter both question and reference answer.")
        else:
            answer, scores = medical_chatbot_with_accuracy(
                question,
                reference
            )

            st.subheader("Generated Answer")
            st.write(answer)

            st.subheader("ROUGE Scores")
            st.write(f"ROUGE-1: {scores['rouge1'].fmeasure:.2f}")
            st.write(f"ROUGE-L: {scores['rougeL'].fmeasure:.2f}")

# ---------------- IMAGE MODE ----------------
else:
    st.subheader("Upload Symptom Image")

    img_file = st.file_uploader(
        "Upload symptom image",
        type=["jpg", "png", "jpeg"]
    )

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        payload = {
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "messages": [
                {"role": "system", "content": "Analyze the image medically."},
                {
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }]
                }
            ],
            "max_tokens": 512
        }

        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }

        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        st.subheader("Medical Analysis")
        st.write(res.json()["choices"][0]["message"]["content"])
