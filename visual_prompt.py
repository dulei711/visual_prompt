import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_gan_model():
    # Load the pre-trained GAN model from torch.hub
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                           'PGAN', model_name='celebAHQ-512',
                           pretrained=True, useGPU=torch.cuda.is_available())
    model = model.cuda()
    model.eval()
    return model

def generate_images(prompt, model, truncation=0.4, truncation_mean=False):
    # Encode the text prompt as an input tensor
    text = F.pad(torch.tensor(model.c_classes).view(1, -1), (0, 0, 0, 128 - model.c_classes.shape[0])).cuda()
    with torch.no_grad():
        noise, _ = model.buildNoiseData(1, 512)
        out = model.test([noise], text, truncation=truncation, truncation_mean=truncation_mean)[0]
    # Convert the output tensor to an image
    out = out.cpu()
    out = (out + 1) / 2
    out = transforms.ToPILImage()(out.squeeze())
    return out

# Load the GAN model
model = load_gan_model()

# Set up the Streamlit app
st.title("GAN Image Generator")
prompt = st.text_input("Enter your text prompt")
if st.button("Send"):
    image = generate_images(prompt, model)
    st.image(image, width=512)
