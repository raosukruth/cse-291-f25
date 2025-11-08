import torch
import os
from diffusers import StableDiffusionInpaintPipeline, StableDiffusion3Pipeline
from transformers import SiglipProcessor, SiglipModel, BlipProcessor, BlipForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from qwen_vl_utils import process_vision_info
from huggingface_hub import login


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

COMMON_DIR = "/home/sjrao/cse291/models"

SD_1_5_inpainting = StableDiffusionInpaintPipeline.from_pretrained("sd-legacy/stable-diffusion-inpainting", torch_dtype=torch.float16).to(device)
SD_1_5_inpainting.save_pretrained(COMMON_DIR + "/sd_1_5_inpainting")
del SD_1_5_inpainting
torch.cuda.empty_cache()

SD_2_inpainting = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16).to(device)
SD_2_inpainting.save_pretrained(COMMON_DIR + "/sd_2_inpainting")
del SD_2_inpainting
torch.cuda.empty_cache()

# SD_3_inpainting = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(device)
# SD_3_inpainting.save_pretrained(COMMON_DIR + "/sd_3_inpainting")
# del SD_3_inpainting
# torch.cuda.empty_cache()

blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
blip_proc.save_pretrained(COMMON_DIR + "/blip_proc")
blip.save_pretrained(COMMON_DIR + "/blip")
del blip_proc
del blip
torch.cuda.empty_cache()

qwen_proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct").to(device)
qwen_proc.save_pretrained(COMMON_DIR + "/qwen_proc")
qwen.save_pretrained(COMMON_DIR + "/qwen")
del qwen_proc
del qwen
torch.cuda.empty_cache()

llava_proc = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
llava =  LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16,).to(device)
llava_proc.save_pretrained(COMMON_DIR + "/llava_proc")
llava.save_pretrained(COMMON_DIR + "/llava")
del llava_proc
del llava
torch.cuda.empty_cache()

sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sbert.save(COMMON_DIR + "/sbert")
del sbert
torch.cuda.empty_cache()

simcse_sup_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
simcse_sup = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
simcse_sup_tokenizer.save_pretrained(COMMON_DIR + "/simcse_sup_tokenizer")
simcse_sup.save_pretrained(COMMON_DIR + "/simcse_sup")
del simcse_sup_tokenizer
del simcse_sup
torch.cuda.empty_cache()

simcse_unsup_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
simcse_unsup = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
simcse_unsup_tokenizer.save_pretrained(COMMON_DIR + "/simcse_unsup_tokenizer")
simcse_unsup.save_pretrained(COMMON_DIR + "/simcse_unsup")
del simcse_unsup_tokenizer
del simcse_unsup
torch.cuda.empty_cache()