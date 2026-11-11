
"""
Women's Health Chatbot - Streamlit App
=======================================
Run base_gpt_finetuned.py first to train and save the model.
Then run this app:

    streamlit run app.py
"""

import torch
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# ======================================================================
# PAGE SETUP
# ======================================================================

st.set_page_config(page_title="Women's Health Chatbot", page_icon="🌸")

st.markdown("""
<style>
.stApp { background-color: #ffe4f0; }
.stTextInput > div > div > input { background-color: white; }
.stButton > button {
    background-color: #ff69b4;
    color: white;
    border: none;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# CUDA
# ======================================================================

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "trained_model"

# ======================================================================
# GENERATION
# ======================================================================

@torch.no_grad()
def generate_response(model,
                      tokenizer,
                      question, 
                      max_new_tokens=150, 
                      temperature=0.7):
                      
    """Generate a response to a user question."""
    model.eval()

    prompt     = f"User: {question}\nAssistant:"
    input_ids  = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

     # Decode only the new tokens (skip the prompt)
    new_tokens = output_ids[0, input_ids.shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

     # Clean up: stop at "User:" if the model tries to continue the conversation
    for stop in ["User:", "\nUser", "\n\nUser", "Assistant:"]:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()

# ======================================================================
# LOAD MODEL
# ======================================================================

if "model" not in st.session_state:
    if not os.path.exists(MODEL_PATH):
        st.error("No trained model found! Please run base_gpt_finetuned.py first to train the model.")
        st.stop()
    with st.spinner("Loading model..."):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
    st.session_state.model     = model
    st.session_state.tokenizer = tokenizer

model     = st.session_state.model
tokenizer = st.session_state.tokenizer

# ======================================================================
# CHAT UI
# ======================================================================

st.title("🌸 Women's Health Chatbot")
st.write("Ask about periods, menstrual phases and general women's health topics.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

temp = st.slider("Temperature (0.1 = focused, 1.5 = creative)", 0.1, 1.5, 0.7, 0.05)

for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🌸 Bot"
    st.write(f"**{role}:** {msg['content']}")

st.divider()

user_input = st.text_input("You:", placeholder="Type your question here...")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Send") and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking..."):
            response = generate_response(model, tokenizer, user_input.strip(), temperature=temp)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()