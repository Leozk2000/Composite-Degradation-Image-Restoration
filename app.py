import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.utils import load_restore_ckpt, load_embedder_ckpt
import os
from gradio_imageslider import ImageSlider

# Enforce CPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define two different model checkpoint paths
embedder_model_path = "ckpts/embedder_model.tar"  # Update with actual path to embedder checkpoint
restorer_model_path_1 = "ckpts/onerestore_model.tar"   # GitHub provided loss

# Load models on CPU only
embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=embedder_model_path)
restorer_1 = load_restore_ckpt(device, freeze_model=True, ckpt_name=restorer_model_path_1)

# Define image preprocessing and postprocessing
transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

def process_image(image, max_size=1024):
    """Resizes the input image if its dimensions exceed the maximum allowed size."""
    w, h = image.size
    should_resize = max(w, h) > max_size
    scale = max_size / max(w, h) if should_resize else 1

    if should_resize:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image

def postprocess_image(tensor):
    image = tensor.squeeze(0).cpu().detach().numpy()
    image = (image) * 255  # Assuming output in [-1, 1], rescale to [0, 255]
    image = np.clip(image, 0, 255).astype("uint8")  # Clip values to [0, 255]
    return Image.fromarray(image.transpose(1, 2, 0))  # Reorder to (H, W, C)


def enhance_image(image, degradation_type=None):
    # Resize image if necessary
    image = process_image(image, max_size=1024)
    
    # Preprocess the image
    input_tensor = torch.Tensor((np.array(image) / 255).transpose(2, 0, 1)).unsqueeze(0).to(device)
    lq_em = transform_resize(image).unsqueeze(0).to(device)
    
    # Handle automatic degradation type estimation
    if degradation_type == "auto" or degradation_type is None:
        # Call the embedder function with type='image_encoder'
        embedder_output = embedder(lq_em, 'image_encoder')
        
        # Parse embedder output
        text_embedding, num_type, text_type = embedder_output
        
        # Handle "clear" predictions
        if "clear" in text_type:
            clear_index = text_type.index("clear")
            print(f"Warning: Predicted 'clear' for image. Using fallback degradation type.")
            
            # Fallback: Use the next most probable type
            if len(text_type) > 1:
                fallback_index = (clear_index + 1) % len(text_type)  # Choose the next type cyclically
                predicted_type = text_type[fallback_index]
            else:
                # If only "clear" exists, default to the first degradation type
                predicted_type = "blur"
        else:
            # Use the predicted type directly
            predicted_type = text_type[0]

        text = predicted_type
    else:
        # Use manually selected degradation type
        embedder_output = embedder([degradation_type], 'text_encoder')
        text_embedding, _, [text] = embedder_output
    
    # Model inference
    with torch.no_grad():
        enhanced_tensor_1 = restorer_1(input_tensor, text_embedding)
    
    # Postprocess outputs
    image_output = postprocess_image(enhanced_tensor_1)
    return (image, image_output), text

# Define the Gradio interface
def inference(image, degradation_type=None):
    return enhance_image(image, degradation_type)

#### Image,Prompts examples
examples = [
            ['image/1_in.png'],
            ['image/2_in.png'],
            ['image/3_in.png'],
            ['image/4_in.png'],
            ['image/5_in.png'],
            ['image/6_in.JPG'],
            ['image/7_in.JPG']
            ]

# Create the Gradio app interface using updated API
interface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="pil", value="image/1_in.png"),  # Image input
        gr.Dropdown(['auto', 'blur', 'blur_haze', 'low', 'haze', 'rain', 'snow',\
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow'], label="Degradation Type", value="auto")  # Manual or auto degradation
    ],
    outputs=[
        ImageSlider(label="Restored Image", 
                   type="pil",
                   show_download_button=True,
                   ),  # Enhanced image output from first model
        gr.Textbox(label="Degradation Type")  # Display the estimated degradation type
    ],
    title="Composite Image Restoration with our improved OneRestore",
    description="Upload an image and enhance it using our OneRestore model modified for traffic scenarios. You can choose to let the model automatically estimate the degradation type or set it manually.",
    examples=examples,
)

# Launch the app
if __name__ == "__main__":
    interface.launch()