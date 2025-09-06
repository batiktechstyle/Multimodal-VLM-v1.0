import os
import time
import threading
import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Glm4vForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from qwen_vl_utils import process_vision_info

# Constants for text generation
MAX_MAX_NEW_TOKENS = 16384
DEFAULT_MAX_NEW_TOKENS = 8192
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Camel-Doc-OCR-062825
MODEL_ID_M = "prithivMLmods/Camel-Doc-OCR-062825"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Megalodon-OCR-Sync-0713
MODEL_ID_T = "prithivMLmods/Megalodon-OCR-Sync-0713"
processor_t = AutoProcessor.from_pretrained(MODEL_ID_T, trust_remote_code=True)
model_t = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_T,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Video-MTR
MODEL_ID_S = "Phoebe13/Video-MTR"
processor_s = AutoProcessor.from_pretrained(MODEL_ID_S, trust_remote_code=True)
model_s = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_S,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load ViLaSR
MODEL_ID_Y = "inclusionAI/ViLaSR"
processor_y = AutoProcessor.from_pretrained(MODEL_ID_Y, trust_remote_code=True)
model_y = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Y,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

def downsample_video(video_path):
    """
    Downsample a video to evenly spaced frames, returning each as a PIL image with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU(duration=120)
def generate_image(model_name: str, text: str, image: Image.Image,
                  max_new_tokens: int = 1024,
                  temperature: float = 0.6,
                  top_p: float = 0.9,
                  top_k: int = 50,
                  repetition_penalty: float = 1.2):
    """
    Generate responses using the selected model for image input.
    """
    if model_name == "Camel-Doc-OCR-062825":
        processor = processor_m
        model = model_m
    elif model_name == "Megalodon-OCR-Sync-0713":
        processor = processor_t
        model = model_t
    elif model_name == "Video-MTR":
        processor = processor_s
        model = model_s
    elif model_name == "ViLaSR-7B":
        processor = processor_y
        model = model_y
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                  max_new_tokens: int = 1024,
                  temperature: float = 0.6,
                  top_p: float = 0.9,
                  top_k: int = 50,
                  repetition_penalty: float = 1.2):
    """
    Generate responses using the selected model for video input.
    """
    if model_name == "Camel-Doc-OCR-062825":
        processor = processor_m
        model = model_m
    elif model_name == "Megalodon-OCR-Sync-0713":
        processor = processor_t
        model = model_t
    elif model_name == "Video-MTR":
        processor = processor_s
        model = model_s
    elif model_name == "ViLaSR-7B":
        processor = processor_y
        model = model_y
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

# Define examples for image and video inference
image_examples = [
    ["convert this page to doc [text] precisely for markdown.", "images/1.png"],
    ["explain the movie shot in detail.", "images/5.jpg"],
    ["convert this page to doc [table] precisely for markdown.", "images/2.png"],
    ["explain the movie shot in detail.", "images/3.png"],
    ["fill the correct numbers.", "images/4.png"]
]

video_examples = [
    ["explain the video in detail.", "videos/b.mp4"],
    ["explain the ad video in detail.", "videos/a.mp4"]
]

# Updated CSS with model choice highlighting
css = """
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
.canvas-output {
    border: 2px solid #4682B4;
    border-radius: 10px;
    padding: 20px;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[Multimodal VLM v1.0](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="✦︎ Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=image_examples,
                        inputs=[image_query, image_upload]
                    )
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="✦︎ Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=video_examples,
                        inputs=[video_query, video_upload]
                    )
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
        with gr.Column():
            with gr.Column(elem_classes="canvas-output"):
                gr.Markdown("## Output")
                output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=3)
                with gr.Accordion("(Result.md)", open=False):
                    markdown_output = gr.Markdown(label="(Result.md)")
            model_choice = gr.Radio(
                choices=["Camel-Doc-OCR-062825", "Video-MTR", "Megalodon-OCR-Sync-0713", "ViLaSR-7B"],
                label="Select Model",
                value="Camel-Doc-OCR-062825"
            )
            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/Multimodal-VLM-v1.0/discussions)")

            gr.Markdown("> [Camel-Doc-OCR-062825](https://huggingface.co/prithivMLmods/Camel-Doc-OCR-062825) is a Qwen2.5-VL-7B-Instruct finetune, highly optimized for document retrieval, structured extraction, analysis, and direct Markdown generation from images and PDFs.")
            gr.Markdown("> [Megalodon-OCR-Sync-0713](https://huggingface.co/prithivMLmods/Megalodon-OCR-Sync-0713), finetuned from Qwen2.5-VL-3B-Instruct, specializes in context-aware multimodal document extraction and analysis, excelling at retrieval, layout parsing, math, and chart/table recognition.")
            gr.Markdown("> [ViLaSR-7B](https://huggingface.co/inclusionAI/ViLaSR) focuses on reinforcing spatial reasoning in visual-language tasks by combining interwoven thinking with visual drawing, making it especially suited for spatial reasoning and complex tip-based queries.")
            gr.Markdown("> [Video-MTR](https://huggingface.co/Phoebe13/Video-MTR) introduces reinforced multi-turn reasoning for long-form video understanding, enabling iterative key segment selection and deeper question comprehension.")
       
            gr.Markdown("> ✋ ViLaSR-7B - demo only supports text-only reasoning, which doesn't reflect the full behavior of the model and may underrepresent its capabilities.")            
            gr.Markdown("> ⚠️ Note: Models in this space may not perform well on video inference tasks.")
    # Define the submit button actions
    image_submit.click(fn=generate_image,
                       inputs=[
                           model_choice, image_query, image_upload,
                           max_new_tokens, temperature, top_p, top_k,
                           repetition_penalty
                       ],
                       outputs=[output, markdown_output])
    video_submit.click(fn=generate_video,
                       inputs=[
                           model_choice, video_query, video_upload,
                           max_new_tokens, temperature, top_p, top_k,
                           repetition_penalty
                       ],
                       outputs=[output, markdown_output])

if __name__ == "__main__":
    demo.queue(max_size=40).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)