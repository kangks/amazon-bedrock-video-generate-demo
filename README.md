# Amazon Bedrock Video Generate Demo

A Streamlit-based demonstration application that showcases video generation capabilities using Amazon Bedrock's Stable Diffusion XL (SDXL) model. This application generates sequences of images based on text prompts and combines them into videos.

## Overview

This demo application demonstrates how to:
- Generate image sequences using Amazon Bedrock's SDXL model
- Convert text prompts into visual sequences
- Create videos from generated image sequences
- Provide an interactive interface for parameter customization

## Features

- **Text-to-Video Generation**: Convert text descriptions into video sequences
- **Customizable Parameters**: 
  - Image dimensions
  - Frame count
  - Generation settings
  - Random seed control
- **Interactive UI**: User-friendly Streamlit interface
- **Image Sequence Control**: View and manage generated image sequences
- **Video Export**: Automatic compilation of images into video format

## Prerequisites

- Python 3.7+
- AWS Account with Amazon Bedrock access
- FFmpeg installed on your system
- AWS credentials configured with appropriate permissions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kangks/amazon-bedrock-video-generate-demo.git
cd amazon-bedrock-video-generate-demo
```

2. Install the required dependencies:
```bash
pip install streamlit boto3 pillow
```

3. Ensure FFmpeg is installed on your system:

  * For Ubuntu/Debian: sudo apt-get install ffmpeg

  * For macOS: brew install ffmpeg

  * For Windows: Download from FFmpeg official website

4. Configure AWS credentials with appropriate permissions for Amazon Bedrock

## Usage

1. Start the Streamlit application:
```bash
streamlit run bedrock_sdxl_sequence_video.py
```

2. In the web interface:
   - Enter your prompt
   - Adjust any desired parameters
   - Click to generate the image sequence
   - The application will create a video from the generated images

## How It Works

The application works in several steps:

1. User inputs a prompt and parameters through the Streamlit interface
2. The system generates a sequence of images using Amazon Bedrock's SDXL model
3. Images are processed and saved locally
4. FFmpeg combines the images into a video file
5. The final video is displayed in the Streamlit interface

## Configuration

The application allows customization of:
- Image dimensions
- Prompts
- Generation parameters
- Random seeds for reproducibility

## Project Structure

├── bedrock_sdxl_sequence_video.py   # Main application file
├── requirements.txt                 # Python dependencies
├── output/                         # Generated images and videos
└── README.md                       # Documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
* Amazon Bedrock for providing the SDXL model
* Streamlit for the web interface framework
* FFmpeg for video processing capabilities

## Disclaimer
This is a demonstration project that uses Amazon Bedrock services. Please be aware of associated AWS costs when running this application.