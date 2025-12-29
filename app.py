import streamlit as st
from PIL import Image
import io
from transformers import pipeline

st.title("AI Story Generator")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    # Display image
    image = Image.open(uploaded_image)
    st.image(image, caption="Your Image", width=300)
    
    if st.button("Generate Story"):
        # Step 1: Image to Text
        with st.spinner("Analyzing image..."):
            image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
            description = image_to_text(image)[0]['generated_text']
        
        st.write(f"**Image description:** {description}")
        
        # Step 2: Generate Story
        with st.spinner("Creating story..."):
            text_generation = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
            story = text_generation(f"Write a short story for kids about: {description}")[0]['generated_text']
        
        st.write(f"**Story:** {story}")
        
        # Step 3: Text to Speech
        with st.spinner("Generating audio..."):
            text_to_speech = pipeline("text-to-speech", model="facebook/mms-tts-eng")
            audio = text_to_speech(story)
            
            # Play audio
            st.audio(audio["audio"], sample_rate=audio["sampling_rate"])
