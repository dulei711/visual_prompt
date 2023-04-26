import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained BigGAN model from GitHub
model = torch.hub.load('huggingface/pytorch-pretrained-BigGAN', 'biggan-deep-512', pretrained=True)

# Define a function to generate an image from a user prompt
def generate_image(prompt):
    # Convert the prompt to a tensor
    prompt_tensor = torch.tensor(model.encode(prompt)).unsqueeze(0)
    # Generate the image using the BigGAN model
    with torch.no_grad():
        image = model.generate_images(prompt_tensor).squeeze(0)
    # Convert the image tensor to a PIL image
    image = transforms.ToPILImage()(image)
    return image

# Define the Streamlit app
def app():
    st.title('Image Generator')
    # Allow the user to input a text prompt
    prompt = st.text_input('Enter a text prompt')
    if prompt:
        # Generate the image from the text prompt
        image = generate_image(prompt)
        # Display the generated image
        st.image(image, caption='Generated Image', use_column_width=True)

# Run app
app()
