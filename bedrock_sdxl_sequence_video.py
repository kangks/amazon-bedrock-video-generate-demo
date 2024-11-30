import streamlit as st
import boto3
import json
import base64
from PIL import Image
import io
import subprocess
from pathlib import Path
import random

class SdxlCfg:
    """
    Class for handling image to image request parameters.
    """

    def __init__(
        self,
        image_width,
        image_height,
        positive_prompt,
        init_image_mode="IMAGE_STRENGTH",
        image_strength=0.5,
        cfg_scale=7,
        clip_guidance_preset="SLOWER",
        sampler="K_DPMPP_2M",
        samples=1,
        steps=30,
        style_preset="photographic",
        extras=None,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.positive_prompt = positive_prompt
        self.init_image_mode = init_image_mode
        self.image_strength = image_strength
        self.cfg_scale = cfg_scale
        self.clip_guidance_preset = clip_guidance_preset
        self.sampler = sampler
        self.samples = samples
        self.steps = steps
        self.style_preset = style_preset
        self.extras = extras

# Initialize Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Replace with your region
)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)

def encode_image(image):
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return f"image/{image.format.lower()}", base64.b64encode(img_bytes).decode('utf-8')

def text_to_image(prompt, output_path=None, seed=None, sdxlCfg=None, model="sdxl"):
    if model == "sdxl":
        # Prepare the request payload for SDXL
        request_body = {
            "text_prompts": [{
                "text": sdxlCfg.positive_prompt,
                "weight": 1
            }],
            "seed": seed,
            "image_strength": sdxlCfg.image_strength,
            "cfg_scale": sdxlCfg.cfg_scale,
            "clip_guidance_preset": sdxlCfg.clip_guidance_preset,
            "sampler": sdxlCfg.sampler,
            "samples": sdxlCfg.samples,
            "steps": sdxlCfg.steps,
            "style_preset": sdxlCfg.style_preset
        }
        model_id = "stability.stable-diffusion-xl-v1"
    else:  # titan
        # Prepare the request payload for Titan Image Generator G1
        request_body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": sdxlCfg.positive_prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": sdxlCfg.samples,
                "quality": "standard",
                "cfgScale": sdxlCfg.cfg_scale,
                "seed": seed if seed is not None else random.randint(0, 2147483647)
            }
        }
        model_id = "amazon.titan-image-generator-v1"

    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    response_body = json.loads(response['body'].read())
    
    if model == "sdxl":
        image_data = base64.b64decode(response_body["artifacts"][0]["base64"])
    else:  # titan
        image_data = base64.b64decode(response_body["images"][0])
        
    image = Image.open(io.BytesIO(image_data))
    if output_path:
        image.save(output_path)
    return image

def image_base64_encoder(image_name):
    """
    This function takes in a string that represent the path to the image that has been uploaded by the user and the function
    is used to encode the image to base64. The base64 string is then returned.
    :param image_name: This is the path to the image file that the user has uploaded.
    :return: A base64 string of the image that was uploaded.
    """
    # opening the image file that was uploaded by the user
    open_image = Image.open(image_name)
    # creating a BytesIO object to store the image in memory
    image_bytes = io.BytesIO()
    # saving the image to the BytesIO object
    open_image.save(image_bytes, format=open_image.format)
    # converting the BytesIO object to a base64 string and returning it
    image_bytes = image_bytes.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    # getting the appropriate file type as claude 3 expects the file type to be presented
    file_type = f"image/{open_image.format.lower()}"
    # returning both the formatted file type string, along with the base64 encoded image
    return file_type, image_base64

def get_image_description(image_path):

    file_type, image_base64 = image_base64_encoder(image_path)
    
    prompt = "Please describe this image in detail to be used as a prompt for an image generation AI. Focus on the visual elements, style, and composition."
    
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": file_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    bedrock = boto3.client('bedrock-runtime')
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        body=json.dumps(payload)
    )
    
    response_body = json.loads(response['body'].read().decode('utf-8'))
    return response_body['content'][0]['text']

def image_to_image(prompt, output_path=None, seed=None, model="sdxl", input_image=None, sdxlCfg=None):
    if seed is None:
        seed = random.randint(0, 2147483647)

    with open(f"{input_image}", "rb") as image_file:
        init_image = base64.b64encode(image_file.read()).decode("utf8")

    if model == "sdxl":
        # Prepare the request payload for SDXL
        request_body = {
            "text_prompts": [{
                "text": sdxlCfg.positive_prompt,
                "weight": 1
            }],
            "init_image": init_image,
            "init_image_mode": sdxlCfg.init_image_mode,
            "image_strength": sdxlCfg.image_strength,
            "cfg_scale": sdxlCfg.cfg_scale,
            "clip_guidance_preset": sdxlCfg.clip_guidance_preset,
            "sampler": sdxlCfg.sampler,
            "samples": sdxlCfg.samples,
            "steps": sdxlCfg.steps,
            "style_preset": sdxlCfg.style_preset,
        }
        model_id = "stability.stable-diffusion-xl-v1"
    else:  # titan
        # Prepare the request payload for Titan Image Generator G1
        request_body = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": sdxlCfg.positive_prompt,
                "images": [init_image]
            },
            "imageGenerationConfig": {
                "numberOfImages": sdxlCfg.samples,
                "quality": "standard",
                "cfgScale": sdxlCfg.cfg_scale,
                "seed": seed
            }
        }
        model_id = "amazon.titan-image-generator-v1"

    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    response_body = json.loads(response['body'].read())
    
    if model == "sdxl":
        image_data = base64.b64decode(response_body["artifacts"][0]["base64"])
    else:  # titan
        image_data = base64.b64decode(response_body["images"][0])
        
    image = Image.open(io.BytesIO(image_data))
    if output_path:
        image.save(output_path)
    return image

def generate_image_sequence(prompt):
    image_files = []
    model = st.session_state.model if 'model' in st.session_state else "sdxl"
    num_variations = st.session_state.model if 'num_variations' in st.session_state else 3

    sdxlCfg = SdxlCfg(
        image_width=512,
        image_height=1024,
        positive_prompt=prompt,
        init_image_mode="IMAGE_STRENGTH",
        image_strength=0.35,
        cfg_scale=7,
        clip_guidance_preset="NONE",
        sampler="K_DPMPP_2M",
        samples=1,
        # seed=seed,
        steps=25,
        style_preset="photographic",
        extras=None,
    )

    with st.spinner('Generating image sequence...'):
        # Generate first image with Titan model
        st.write("Generating initial image...")
        first_image_path = output_dir / "image_000.png"
        first_image_seed = random.randint(0, 2147483647)
        text_to_image(prompt, str(first_image_path), seed=first_image_seed, sdxlCfg=sdxlCfg, model=model)
        image_files.append(str(first_image_path))
        st.image(str(first_image_path))
        
        # Generate variations using SDXL
        for i in range(num_variations):
            st.write(f"Generating variation {i+1}/{num_variations}...")
            
            # Generate variation with a different seed
            variation_image_seed = random.randint(0, 2147483647)
            image_path = output_dir / f"image_{i+1:03d}.png"
            
            # Generate variation using SDXL
            image_to_image(
                prompt, 
                str(image_path), 
                seed=variation_image_seed,
                model=model,
                input_image=str(first_image_path),
                sdxlCfg=sdxlCfg
            )
            image_files.append(str(image_path))
            
            # Display the generated variation
            st.image(str(image_path))
    
    return image_files

def create_video(image_files, output_path):
    with st.spinner('Creating video...'):
        # FFmpeg command to create video from images
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-framerate', '9',  # frame per second
            '-pattern_type', 'sequence',
            '-i', str(output_dir / 'image_%03d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            st.success('Video created successfully!')
        except subprocess.CalledProcessError as e:
            st.error(f'Error creating video: {e}')

# Streamlit UI
st.title("Amazon Bedrock Multimodal Demo")

# Add instructions in the sidebar
tab1, tab2 = st.tabs(["Image Sequence", "Image Variations"])

with tab1:
    st.header("Generate Image Sequence")
    sequence_prompt = st.text_area("Enter your image sequence description:", 
                                value="A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape.")

    if st.button("Generate Image Sequence"):
        if sequence_prompt:

            random.seed( 123 )
            # Generate image sequence
            image_files = generate_image_sequence(sequence_prompt)
            
            # Create video from generated images
            video_path = output_dir / "output.mp4"
            create_video(image_files, video_path)
            
            # Display video
            if video_path.exists():
                st.video(str(video_path))
        else:
            st.warning("Please enter a prompt first.")
                
with tab2:
    st.header("Generate Image Variations")
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], key="variation_image")
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Variations"):
            # Save uploaded file temporarily
            temp_path = f'{temp_dir}/"temp_upload.jpg"'
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Get image description from Claude
            with st.spinner('Getting image description...'):
                description = get_image_description(temp_path)
                st.write("Generated Description:")
                st.write(description)
            
            random.seed( 456 )

            # Generate variations using the description
            with st.spinner('Generating variations...'):
                image_files = generate_image_sequence(description)

            # Create video from generated images
            video_path = output_dir / "output.mp4"
            create_video(image_files, video_path)
            
            # Display video
            if video_path.exists():
                st.video(str(video_path))


with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Image Sequence Tab:
       - Enter a detailed description
       - Click 'Generate Image Sequence' to create images sequence and video
       
    2. Image Variations Tab:
       - Upload an image
       - Click 'Generate Variations' to get AI-generated description and variations
    """)
    
    model = st.selectbox(
        "Select Model",
        options=["sdxl", "titan"],
        help="Choose between Stable Diffusion XL and Amazon Titan models",
        key="model"
    )

    num_variations = st.slider("Number of variations", min_value=1, max_value=10, value=3)
    
    st.info("Note: Make sure you have proper AWS credentials configured with Bedrock access.")