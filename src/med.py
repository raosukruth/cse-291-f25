import nltk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from diffusers import StableDiffusionInpaintPipeline, StableDiffusion3InpaintPipeline
import torch
from PIL import Image
from pathlib import Path
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import kagglehub
from transformers import SiglipProcessor, SiglipModel, BlipProcessor, BlipForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration
from qwen_vl_utils import process_vision_info
from nltk.translate.bleu_score import sentence_bleu,  SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import re
from rouge_score import rouge_scorer
from IPython.display import display 

device = "cpu"
if torch.cuda.is_available():
  device = "cuda"

try:
	nltk.data.find('corpora/wordnet')
except LookupError:
	nltk.download('wordnet', quiet=True)

try:
	nltk.data.find('corpora/omw-1.4')
except LookupError:
	nltk.download('omw-1.4', quiet=True)

BASE_DIR = "/home/sjrao/cse291"
DATA_DIR = BASE_DIR + "/data"
MODELS_DIR = BASE_DIR + "/models"
IMAGE_DIR = DATA_DIR + "/archive/images/images_normalized"
OUTPUTS_DIR = BASE_DIR + "/outputs"
ARCHIVE_S_DIR = BASE_DIR + "/archive"
ORIGINAL_MASK_DIR = OUTPUTS_DIR + "/original_masked_outputs"
INPAINTING_OUTPUTS_DIR = OUTPUTS_DIR + "/inpainting_outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ORIGINAL_MASK_DIR, exist_ok=True)
os.makedirs(INPAINTING_OUTPUTS_DIR, exist_ok=True)
os.makedirs(INPAINTING_OUTPUTS_DIR + "/sd_1_5_outputs", exist_ok=True)
os.makedirs(INPAINTING_OUTPUTS_DIR + "/sd_2_outputs", exist_ok=True)
os.makedirs(INPAINTING_OUTPUTS_DIR + "/sd_3_outputs", exist_ok=True)

def color_quantize(patch, k=8):
    Z = patch.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(patch.shape)

def jpeg_artifact_patch(patch, quality=10):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    degraded = cv2.imread(tmp.name)
    return cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

def get_distorted_center(original_image_path, save_original=False, save_original_path="outputs/original.png", save_modified=False, save_modified_path="outputs/center_mask.png", visualize_result=False):
  imh = cv2.imread(original_image_path)
  imh = cv2.cvtColor(imh, cv2.COLOR_BGR2RGB)

  gray = cv2.cvtColor(imh, cv2.COLOR_RGB2GRAY)
  _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
  coords = cv2.findNonZero(thresh)
  x, y, w, h = cv2.boundingRect(coords)
  imh = imh[y:y+h, x:x+w]
  imh = cv2.resize(imh, (128, 128))

  if save_original:
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(save_original_path, cv2.cvtColor(imh, cv2.COLOR_RGB2BGR))

  h, w, _ = imh.shape
  ch, cw = 64, 64
  y1, y2 = h//2 - ch//2, h//2 + ch//2
  x1, x2 = w//2 - cw//2, w//2 + cw//2

  center_mask = imh.copy()
  center_mask[y1:y2, x1:x2] = 0
  if save_modified:
    cv2.imwrite(save_modified_path, cv2.cvtColor(center_mask, cv2.COLOR_RGB2BGR))

  if visualize_result:
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1); plt.imshow(imh); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(center_mask); plt.title("Center Mask")
    plt.tight_layout()
    plt.show()

  return center_mask

def get_distorted_gaussian(original_image_path, save_original=False, save_original_path="outputs/original.png", save_modified=False, save_modified_path="outputs/gaussian_center.png", visualize_result=False):
  imh = cv2.imread(original_image_path)
  imh = cv2.cvtColor(imh, cv2.COLOR_BGR2RGB)

  gray = cv2.cvtColor(imh, cv2.COLOR_RGB2GRAY)
  _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
  coords = cv2.findNonZero(thresh)
  x, y, w, h = cv2.boundingRect(coords)
  imh = imh[y:y+h, x:x+w]
  imh = cv2.resize(imh, (128, 128))

  if save_original:
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(save_original_path, cv2.cvtColor(imh, cv2.COLOR_RGB2BGR))

  h, w, _ = imh.shape
  ch, cw = 64, 64
  y1, y2 = h//2 - ch//2, h//2 + ch//2
  x1, x2 = w//2 - cw//2, w//2 + cw//2

  gaussian_center = imh.copy()
  # gaussian_center[y1:y2, x1:x2] = cv2.GaussianBlur(imh[y1:y2, x1:x2], (9, 9), 0)
  gaussian_center[y1:y2, x1:x2] = cv2.GaussianBlur(imh[y1:y2, x1:x2], (51, 51), 0)

  if save_modified:
    cv2.imwrite(save_modified_path, cv2.cvtColor(gaussian_center, cv2.COLOR_RGB2BGR))

  if visualize_result:
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1); plt.imshow(imh); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(gaussian_center); plt.title("Gaussian")
    plt.tight_layout()
    plt.show()

  return gaussian_center


def low_dim(original_image_path, save_original=False, save_original_path="outputs/original.png", save_modified=False, save_modified_path="outputs/degraded_center.png", visualize_result=False):
  imh = cv2.imread(original_image_path)
  imh = cv2.cvtColor(imh, cv2.COLOR_BGR2RGB)

  gray = cv2.cvtColor(imh, cv2.COLOR_RGB2GRAY)
  _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
  coords = cv2.findNonZero(thresh)
  x, y, w, h = cv2.boundingRect(coords)
  imh = imh[y:y+h, x:x+w]
  imh = cv2.resize(imh, (128, 128))

  if save_original:
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(save_original_path, cv2.cvtColor(imh, cv2.COLOR_RGB2BGR))

  h, w, _ = imh.shape
  ch, cw = 64, 64
  y1, y2 = h//2 - ch//2, h//2 + ch//2
  x1, x2 = w//2 - cw//2, w//2 + cw//2

  degraded = imh.copy()
  patch = imh[y1:y2, x1:x2]
  quantized = color_quantize(patch, k=8)
  degraded_patch = jpeg_artifact_patch(quantized, quality=5)
  alpha = 0.9
  degraded[y1:y2, x1:x2] = cv2.addWeighted(degraded[y1:y2, x1:x2], 1 - alpha, degraded_patch, alpha, 0)
  if save_modified:
    cv2.imwrite(save_modified_path, cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))

  if visualize_result:
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1); plt.imshow(imh); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(degraded); plt.title("Low-Dimensional")
    plt.tight_layout()
    plt.show()

  return degraded


def get_distorted_all(original_image_path, save_original_path, save_modified_center_path, save_modified_gaussian_path, save_modified_degraded_path, visualize_result=False):
  imh = cv2.imread(original_image_path)
  imh = cv2.cvtColor(imh, cv2.COLOR_BGR2RGB)

  gray = cv2.cvtColor(imh, cv2.COLOR_RGB2GRAY)
  _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
  coords = cv2.findNonZero(thresh)
  x, y, w, h = cv2.boundingRect(coords)
  imh = imh[y:y+h, x:x+w]
  imh = cv2.resize(imh, (128, 128))

  if save_original_path is not None:
    save_dir = os.path.dirname(save_original_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_original_path, cv2.cvtColor(imh, cv2.COLOR_RGB2BGR))

  h, w, _ = imh.shape
  ch, cw = 64, 64
  y1, y2 = h//2 - ch//2, h//2 + ch//2
  x1, x2 = w//2 - cw//2, w//2 + cw//2

  center_mask = imh.copy()
  center_mask[y1:y2, x1:x2] = 0

  gaussian_center = imh.copy()
  gaussian_center[y1:y2, x1:x2] = cv2.GaussianBlur(imh[y1:y2, x1:x2], (9, 9), 0)

  degraded = imh.copy()
  patch = imh[y1:y2, x1:x2]
  quantized = color_quantize(patch, k=8)
  degraded_patch = jpeg_artifact_patch(quantized, quality=5)
  alpha = 0.9
  degraded[y1:y2, x1:x2] = cv2.addWeighted(degraded[y1:y2, x1:x2], 1 - alpha, degraded_patch, alpha, 0)
  
  if save_modified_center_path is not None and save_modified_gaussian_path is not None and save_modified_degraded_path is not None:
    for p in (save_modified_center_path, save_modified_gaussian_path, save_modified_degraded_path):
      d = os.path.dirname(p) or "."
      os.makedirs(d, exist_ok=True)
    cv2.imwrite(save_modified_center_path, cv2.cvtColor(center_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_modified_gaussian_path, cv2.cvtColor(gaussian_center, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_modified_degraded_path, cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
  if visualize_result:
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1); plt.imshow(imh); plt.title("Original")
    plt.subplot(1, 4, 2); plt.imshow(center_mask); plt.title("Center Mask")
    plt.subplot(1, 4, 3); plt.imshow(gaussian_center); plt.title("Gaussian")
    plt.subplot(1, 4, 4); plt.imshow(degraded); plt.title("Low-Dimensional")
    plt.tight_layout()
    plt.show()

  return imh, center_mask, gaussian_center, degraded

# SD inpainting models - load these in the processing script when needed
# SD_1_5_inpainting = StableDiffusionInpaintPipeline.from_pretrained(MODELS_DIR + "/sd_1_5_inpainting").to(device)
# SD_2_inpainting = StableDiffusionInpaintPipeline.from_pretrained(MODELS_DIR + "/sd_2_inpainting").to(device)

def create_center_mask(h=128, w=128, ch=64, cw=64):
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    y1, y2 = h//2 - ch//2, h//2 + ch//2
    x1, x2 = w//2 - cw//2, w//2 + cw//2
    mask[y1:y2, x1:x2] = 255
    return mask

def inpaint_with_SD_2(model, image, mask_image, prompt, save_path, num_inference_steps=50, guidance_scale=7.5, strength=1.0, visualize_image=False):

  if not isinstance(image, Image.Image):
    image_pil = Image.fromarray(image.astype(np.uint8))
  else:
      image_pil = image
  mask_pil = Image.fromarray(mask_image.astype(np.uint8))
  inpainted_image = model(prompt=prompt, image=image_pil, mask_image=mask_pil, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength).images[0]

  if save_path is not None:
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    inpainted_image.save(save_path)

  if visualize_image:
    plt.imshow(inpainted_image)

  return inpainted_image


def inpaint_with_SD_1_5(model, image, mask_image, prompt, save_path, num_inference_steps=50, guidance_scale=7.5, strength=1.0, visualize_image=False):

  if not isinstance(image, Image.Image):
    image_pil = Image.fromarray(image.astype(np.uint8))
  else:
      image_pil = image
  mask_pil = Image.fromarray(mask_image.astype(np.uint8))
  inpainted_image = model(prompt=prompt, image=image_pil, mask_image=mask_pil, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength).images[0]

  if save_path is not None:
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    inpainted_image.save(save_path)

  if visualize_image:
    plt.imshow(inpainted_image)

  return inpainted_image

# def inpaint_with_SD_3(model, image, mask_image, prompt, save_path, num_inference_steps=50, guidance_scale=7.5, strength=1.0, visualize_image=False):

#   if not isinstance(image, Image.Image):
#     image_pil = Image.fromarray(image.astype(np.uint8))
#   else:
#       image_pil = image
#   mask_pil = Image.fromarray(mask_image.astype(np.uint8))
#   inpainted_image = model(prompt=prompt, image=image_pil, mask_image=mask_pil, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength).images[0]

#   save_dir = os.path.dirname(save_path) or "."
#   os.makedirs(save_dir, exist_ok=True)
#   inpainted_image.save(save_path)

#   if visualize_image:
#     plt.imshow(inpainted_image)

#   return inpainted_image

def blip_caption(image):
  blip_proc = BlipProcessor.from_pretrained(MODELS_DIR + "/blip_proc")
  blip = BlipForConditionalGeneration.from_pretrained(MODELS_DIR + "/blip", torch_dtype=torch.float16).to(device)
  if not isinstance(image, Image.Image):
    pil = Image.fromarray(image.astype(np.uint8))
  else:
    pil = image

  tensors = blip_proc(images=pil, text="Report any abnormalities in the x-ray in an extremely short phrase.",  return_tensors="pt")
  tensors = {k: v.to(device) for k, v in tensors.items()}
  output = blip.generate(**tensors)

  caption = blip_proc.decode(output[0], skip_special_tokens=True)

  del blip
  del blip_proc
  torch.cuda.empty_cache()
  
  return caption

def qwen_caption(image):
  qwen_proc = AutoProcessor.from_pretrained(MODELS_DIR + "/qwen_proc")
  qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS_DIR + "/qwen", torch_dtype=torch.float16, device_map="auto")
  if not isinstance(image, Image.Image):
    pil = Image.fromarray(image.astype(np.uint8))
  else:
    pil = image

  messages = [
    {
      "role": "user",
      "content": [
          {
              "type": "text",
              "text": "Report any abnormalities in the x-ray in an extremely short phrase."
          },
          {
              "type": "image",
              "image": pil
          },
      ],
    }]

  text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

  images, videos = process_vision_info(messages)
  tensors = qwen_proc(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")

  output = qwen.generate(**tensors, max_new_tokens=100)
  answer = qwen_proc.batch_decode(output, skip_special_tokens=True)[0].strip()

  if "assistant" in answer:
      answer = answer.split("assistant")[-1].strip()

  del qwen
  del qwen_proc
  torch.cuda.empty_cache()

  return answer
  

def llava_caption(image):
  llava_proc = AutoProcessor.from_pretrained(MODELS_DIR + "/llava_proc")
  llava =  LlavaOnevisionForConditionalGeneration.from_pretrained(MODELS_DIR + "/llava", torch_dtype=torch.float16).to(device)
  if not isinstance(image, Image.Image):
    pil = Image.fromarray(image.astype(np.uint8))
  else:
    pil = image

  messages = [
      {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Report any abnormalities in the x-ray in an extremely short phrase."
            },
            {
                "type": "image",
                "image": pil
            },
        ],
      }]

  text = llava_proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
  text_cuda = {}
  for k, v in text.items():
    text_cuda[k] = v.to("cuda")
  outputs = llava.generate(**text_cuda, max_new_tokens=100)
  answer = llava_proc.decode(outputs[0], skip_special_tokens=True)
  if "assistant" in answer:
        answer = answer.split("assistant")[-1].strip()

  del llava
  del llava_proc
  torch.cuda.empty_cache()
  
  return answer