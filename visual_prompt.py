import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_gan_model(model_name, truncation=0.4, truncation_mean=False):
    # Load the pre-trained GAN model from torch.hub
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                            'PGAN', model_name=model_name,
                            pretrained=True, useGPU=torch.cuda.is_available())
    model = model.cuda()
    model.eval()
    model.truncation = truncation
    model.truncation_mean = truncation_mean
    return model

def generate_images(prompt, model):
    # Encode the text prompt as an input tensor
    text = F.pad(torch.tensor(model.c_classes).view(1, -1), (0, 0, 0, 128 - model.c_classes.shape[0])).cuda()
    with torch.no_grad():
        noise, _ = model.buildNoiseData(1, 512)
        out = model.test([noise], text)[0]
    # Convert the output tensor to an image
    out = out.cpu()
    out = (out + 1) / 2
    out = transforms.ToPILImage()(out.squeeze())
    return out

# Set up the Streamlit app
st.title("GAN Image Generator")
model_name = st.selectbox("Select a pre-trained GAN model", ["celebAHQ-512", "bedrooms-256", "cars-512"])
truncation = st.slider("Truncation value", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
truncation_mean = st.checkbox("Use truncation mean", value=False)
prompt = st.text_input("Enter your text prompt")

# Load the GAN model
model = load_gan_model(model_name, truncation, truncation_mean)

# Generate the image when the user enters a prompt
if st.button("Run"):
    image = generate_images(prompt, model)
    st.image(image, width=model.output_size)    
