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

BASE_DIR = "/home/sjrao/cse291"
DATA_DIR = BASE_DIR + "/data"
MODELS_DIR = BASE_DIR + "/models"

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


def create_center_mask(h=128, w=128, ch=64, cw=64):
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    y1, y2 = h//2 - ch//2, h//2 + ch//2
    x1, x2 = w//2 - cw//2, w//2 + cw//2
    mask[y1:y2, x1:x2] = 255
    return mask