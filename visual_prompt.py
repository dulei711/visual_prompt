import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained GAN model
model = torch.load('pretrained_model.pth', map_location=torch.device('cpu'))

# Define a function to generate an image from a user prompt
def generate_image(prompt):
    # Convert the prompt to a tensor
    prompt_tensor = transforms.ToTensor()(prompt)
    # Generate the image using the GAN model
    with torch.no_grad():
        image = model(prompt_tensor)
    # Convert the image tensor to a PIL image
    image = transforms.ToPILImage()(image)
    return image

# Define the Streamlit app
def app():
    st.title('Image Generator')
    # Allow the user to upload an image prompt
    prompt = st.file_uploader('Upload an image prompt', type=['jpg', 'jpeg', 'png'])
    if prompt:
        # Generate the image from the user prompt
        image = generate_image(prompt)
        # Display the generated image
        st.image(image, caption='Generated Image', use_column_width=True)

# Run app
app()
