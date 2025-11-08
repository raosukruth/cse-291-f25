import med
import pandas as pd
import os
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
from PIL import Image


def get_images_list(dir_name, number_of_images=2000):
    all_files = os.listdir(dir_name)
    sorted_files = sorted(all_files, key=get_row_number)

    files = []
    count = 0
    last_index = None
    for file_name in sorted_files:
        if count > number_of_images:
            files.pop()
            break
        index = get_row_number(file_name)
        if index:
            if index != last_index:
                count += 1
        last_index = index
        files.append(file_name)  
    return files

def get_row_number(image_name):
    return int(image_name.split('_')[0])
    
def get_csv_row(csv_file, row_number):
    df = pd.read_csv(csv_file)

    if row_number < 0 or row_number >= len(df):
        raise IndexError("Out of range")

    return df.iloc[row_number]

def get_csv_findings(csv_file, row_number):
    row = get_csv_row(csv_file, row_number)
    findings = row['findings']
    if pd.isna(findings):
        if pd.isna(row['impression']):
            return str('No acute cardiopulmonary findings.')
        return str('impression')
    return str(findings)

def get_findings(image_name, csv_file):
    row_number = get_row_number(image_name) - 1
    return get_csv_findings(csv_file, row_number)

def mask_inpaint(image_path, image_name, findings):

    #  Not saving the masked images to save disk space
    imh, center_mask, gaussian_center, degraded = med.get_distorted_all(image_path, None, 
                                                                            None, None,
                                                                             None, visualize_result=False)
    
    mask_image = med.create_center_mask()
    
    SD_1_5 = med.StableDiffusionInpaintPipeline.from_pretrained(med.MODELS_DIR + "/sd_1_5_inpainting").to(med.device)
    torch.cuda.empty_cache()
    SD_2 = med.StableDiffusionInpaintPipeline.from_pretrained(med.MODELS_DIR + "/sd_2_inpainting").to(med.device)
    torch.cuda.empty_cache()
    
    # SD 1.5
    SD_1_5_mask = med.inpaint_with_SD_1_5(SD_1_5, center_mask, mask_image, findings, None)
    SD_1_5_gaussian = med.inpaint_with_SD_1_5(SD_1_5, gaussian_center, mask_image, findings, None)
    SD_1_5_low_dim = med.inpaint_with_SD_1_5(SD_1_5, degraded, mask_image, findings, None)

    # SD 2 
    SD_2_mask = med.inpaint_with_SD_2(SD_2, center_mask, mask_image, findings, None)
    SD_2_gaussian = med.inpaint_with_SD_2(SD_2, gaussian_center, mask_image, findings, None)
    SD_2_low_dim = med.inpaint_with_SD_2(SD_2, degraded, mask_image, findings, None)

    del SD_1_5
    del SD_2
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return [SD_1_5_mask, SD_1_5_gaussian, SD_1_5_low_dim, SD_2_mask, SD_2_gaussian, SD_2_low_dim]

def process_image(image_path, image_name, findings):
    SD_1_5_mask, SD_1_5_gaussian, SD_1_5_low_dim, SD_2_mask, SD_2_gaussian, SD_2_low_dim = mask_inpaint(image_path, image_name, findings)

    all_captions = {}
    # BLIP
    all_captions['SD_1_5_mask_blip'] = med.blip_caption(SD_1_5_mask)
    all_captions['SD_1_5_gaussian_blip'] = med.blip_caption(SD_1_5_gaussian)
    all_captions['SD_1_5_low_dim_blip'] = med.blip_caption(SD_1_5_low_dim)
    all_captions['SD_2_mask_blip'] = med.blip_caption(SD_2_mask)
    all_captions['SD_2_gaussian_blip'] = med.blip_caption(SD_2_gaussian)
    all_captions['SD_2_low_dim_blip'] = med.blip_caption(SD_2_low_dim)

    # Qwen
    all_captions['SD_1_5_mask_qwen'] = med.qwen_caption(SD_1_5_mask)
    all_captions['SD_1_5_gaussian_qwen'] = med.qwen_caption(SD_1_5_gaussian)
    all_captions['SD_1_5_low_dim_qwen'] = med.qwen_caption(SD_1_5_low_dim)
    all_captions['SD_2_mask_qwen'] = med.qwen_caption(SD_2_mask)
    all_captions['SD_2_gaussian_qwen'] = med.qwen_caption(SD_2_gaussian)
    all_captions['SD_2_low_dim_qwen'] = med.qwen_caption(SD_2_low_dim)

    # LLaVA
    all_captions['SD_1_5_mask_llava'] = med.llava_caption(SD_1_5_mask)
    all_captions['SD_1_5_gaussian_llava'] = med.llava_caption(SD_1_5_gaussian)
    all_captions['SD_1_5_low_dim_llava'] = med.llava_caption(SD_1_5_low_dim)
    all_captions['SD_2_mask_llava'] = med.llava_caption(SD_2_mask)
    all_captions['SD_2_gaussian_llava'] = med.llava_caption(SD_2_gaussian)
    all_captions['SD_2_low_dim_llava'] = med.llava_caption(SD_2_low_dim)

    # Generate captions for original image
    original_img = Image.open(image_path)
    all_captions['original_blip'] = med.blip_caption(original_img)
    all_captions['original_qwen'] = med.qwen_caption(original_img)
    all_captions['original_llava'] = med.llava_caption(original_img)

   
    # Errors (B1, B2, B3, B4, METEOR, ROUGE-L, SBERT, Supervised SimCSE, Unsupervised SimCSE)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    sbert = SentenceTransformer(med.MODELS_DIR + '/sbert')

    simcse_sup = AutoModel.from_pretrained(med.MODELS_DIR + '/simcse_sup')
    simcse_sup_tokenizer = AutoTokenizer.from_pretrained(med.MODELS_DIR + '/simcse_sup_tokenizer')

    simcse_unsup = AutoModel.from_pretrained(med.MODELS_DIR + '/simcse_unsup')
    simcse_unsup_tokenizer = AutoTokenizer.from_pretrained(med.MODELS_DIR + '/simcse_unsup_tokenizer')
    
    results = {"image_id": image_name}

    for model, caption in all_captions.items():
        results[model + "_bleu_1"] = sentence_bleu([findings.lower().split()], caption.lower().split(), weights=(1, 0, 0, 0))
        results[model + "_bleu_2"] = sentence_bleu([findings.lower().split()], caption.lower().split(), weights=(0.5, 0.5, 0, 0))
        results[model + "_bleu_3"] = sentence_bleu([findings.lower().split()], caption.lower().split(), weights=(0.33, 0.33, 0.33, 0))
        results[model + "_bleu_4"] = sentence_bleu([findings.lower().split()], caption.lower().split())
        results[model + "_meteor"] = meteor_score([findings.lower().split()], caption.lower().split())
        results[model + "_rouge"] = scorer.score(findings.lower(), caption.lower())['rougeL'].fmeasure
        
        inputs_sup = simcse_sup_tokenizer([findings, caption], padding=True, truncation=True, return_tensors="pt")
        embeddings_sup = simcse_sup(**inputs_sup, output_hidden_states=True, return_dict=True).pooler_output
        results[model + "_simcse_sup"] = 1 - cosine(embeddings_sup[0].detach().cpu().numpy(), embeddings_sup[1].detach().cpu().numpy())
        
        inputs_unsup = simcse_unsup_tokenizer([findings, caption], padding=True, truncation=True, return_tensors="pt")
        embeddings_unsup = simcse_unsup(**inputs_unsup, output_hidden_states=True, return_dict=True).pooler_output
        results[model + "_simcse_unsup"] = 1 - cosine(embeddings_unsup[0].detach().cpu().numpy(), embeddings_unsup[1].detach().cpu().numpy())

        findings_emb = sbert.encode(findings, convert_to_tensor=True)
        caption_emb = sbert.encode(caption, convert_to_tensor=True)
        results[model + "_sbert"] = 1 - cosine(findings_emb.detach().cpu().numpy(), caption_emb.detach().cpu().numpy())
    
    return results

if __name__ == "__main__":
    common_dir = "/home/sjrao/cse291/data/archive/"
    images_dir = common_dir + "images/images_normalized"
    reports_dir = common_dir + "indiana_reports.csv"

    images = get_images_list(images_dir, number_of_images=3)

    output_path = "/home/sjrao/cse291/outputs/score_outputs/results.csv"
    
    for i, image in enumerate(images):
        findings = get_findings(image, reports_dir)
        image_path = images_dir + "/" + image
        result = process_image(image_path, image, findings)

        df = pd.DataFrame([result])
        if i == 0:
            df.to_csv(output_path, mode='w', index=False, header=True)
        else:
            df.to_csv(output_path, mode='a', index=False, header=False)

        print("Image {} finished".format(image))