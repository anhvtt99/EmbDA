import torch
import torch.nn.functional as F
import os
import ijson
from collections import OrderedDict
import numpy as np
from math import lgamma
from sentence_splitter import SentenceSplitter
from sinling import SinhalaTokenizer
from faiss_search import faiss_L2_search, faiss_cos_search


def count_files(folder_path):
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def sbs_split(doc, lang):
    if lang == 'si':
        splitter = SinhalaTokenizer()
        sentences = splitter.split_sentences(doc)
    elif lang == 'ta':
        sentences = doc.split('.')
        sentences = [sent + '.' for sent in sentences if sent != '']
    elif lang == 'en':
        splitter = SentenceSplitter(language=lang)
        sentences = splitter.split(doc)
    return sentences

def pad_list(lst):
    max_length = max(len(sublist) for sublist in lst)
    padded_lst = [sublist + [0] * (max_length - len(sublist)) for sublist in lst]
    return padded_lst

def ofls_split(doc, segment_len, overlap, tokenizer, device, chunk_size=2048):
    'split from start'
    doc_tkn = tokenizer(doc, truncation=False)
    doc_ids = doc_tkn['input_ids'][1:-1]
    doc_token_type_ids = doc_tkn['token_type_ids'][1:-1]
    doc_mask = doc_tkn['attention_mask'][1:-1]
    orig_indices = []
    out_indices = []
    start_idx = 0

    if len(doc_ids) > overlap:
        while start_idx < len(doc_ids) - overlap:
            end_idx = min(start_idx + segment_len, len(doc_ids))
            segment_ids = doc_ids[start_idx:end_idx]
            orig_indices.append(segment_ids)
            start_idx += segment_len - overlap
    else:
        while start_idx < len(doc_ids):
            end_idx = min(start_idx + segment_len, len(doc_ids))
            segment_ids = doc_ids[start_idx:end_idx]
            orig_indices.append(segment_ids)
            start_idx += segment_len - overlap
            
    for sublist in orig_indices:
        re_sublist = [101] + sublist + [102]
        out_indices.append(re_sublist)
    out_token_type_ids = [[0] * len(sublist) for sublist in out_indices]
    out_mask = [[1] * len(sublist) for sublist in out_indices]
    
    out_indices_pad = pad_list(out_indices)
    out_token_type_ids_pad = pad_list(out_token_type_ids)
    out_mask_pad = pad_list(out_mask)
    
    ids_batches = split_list(out_indices_pad, chunk_size)
    token_type_ids_batches = split_list(out_token_type_ids_pad, chunk_size)
    mask_batches = split_list(out_mask_pad, chunk_size)
    
    output_batches = []
    for ids_batch, token_type_ids_batch, mask_batch in zip(ids_batches, token_type_ids_batches, mask_batches):
        output_batch = {
        "input_ids": torch.tensor(ids_batch).to(device),
        "token_type_ids": torch.tensor(token_type_ids_batch).to(device),
        "attention_mask": torch.tensor(mask_batch).to(device)
        }
        output_batches.append(output_batch)  
    
    return orig_indices, output_batches

def ofls_split_tkn(doc, segment_len, overlap, tokenizer):
    doc_tkn = tokenizer.tokenize(doc)
    segments = []
    start_idx = 0
    segments_len = []

    if len(doc_tkn) > overlap:
        while start_idx < len(doc_tkn) - overlap:
            end_idx = min(start_idx + segment_len, len(doc_tkn))
            segment_tokens = doc_tkn[start_idx:end_idx]
            segments_len.append(len(segment_tokens))
            segment = tokenizer.convert_tokens_to_string(segment_tokens)
            segments.append(segment)
            start_idx += segment_len - overlap
    else:
        while start_idx < len(doc_tkn):
            end_idx = min(start_idx + segment_len, len(doc_tkn))
            segment_tokens = doc_tkn[start_idx:end_idx]
            segments_len.append(len(segment_tokens))
            segment = tokenizer.convert_tokens_to_string(segment_tokens)
            segments.append(segment)
            start_idx += segment_len - overlap
    return segments

def cal_LIDF(segments_list):
    segments_list = [[str(inner) for inner in outer]for outer in segments_list]
    final_IDF_list = []
    segment_counts = {}
    
    for segment in segments_list:
        for item in segment:
            if item in segment_counts:
                segment_counts[item] += 1
            else:
                segment_counts[item] = 1
        
    for segment in segments_list:
        sublist_counts = np.array([segment_counts[item] for item in segment])
        sublist_LIDF = 1 / sublist_counts
        sublist_LIDF_sum = np.sum(sublist_LIDF)
        final_IDF_list.append(sublist_LIDF / sublist_LIDF_sum)
    
    return final_IDF_list

def _beta_pdf_unit_interval(x, alpha, beta):
    eps = 1e-8
    x = np.clip(x, eps, 1.0 - eps)

    log_norm = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
    log_pdf = log_norm + (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1.0 - x)
    return np.exp(log_pdf)


def pert_windows(J, N, r):
    if N <= 0:
        raise ValueError("N must be > 0")
    if J <= 0:
        raise ValueError("J must be > 0")

    if N == 1:
        return torch.ones((J, 1), dtype=torch.float32)

    result_list_final = []

    a = -1.0
    c = float(N)
    xs = np.arange(N, dtype=np.float64)

    for j in range(J):
        b = (j + 0.5) / J * N

        u = (xs - a) / (c - a)
        alpha = 1.0 + r * (b - a) / (c - a)
        beta = 1.0 + r * (c - b) / (c - a)

        probs = _beta_pdf_unit_interval(u, alpha, beta)

        probs_sum = probs.sum()
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.ones(N, dtype=np.float64) / N
        else:
            probs = probs / probs_sum

        result_list_final.append(probs.tolist())

    return torch.tensor(result_list_final, dtype=torch.float32)
def load_data(emb_files_num, lang, split_method, embs_path):
    IDs = []
    values = []
    for i in range(emb_files_num):
        print(f"Begin loading [{lang}] data file [{i}]...")
        if split_method == "ofls":
            files_path = embs_path + f'/{i}-emb.json'
        elif split_method == "sbs":
            files_path = embs_path + f'/{i}-emb.json'
        
        with open(files_path, 'r') as f1:
            for doc in ijson.items(f1, 'lr_docs.item', use_float=True):
                id, value = doc.popitem()
                IDs.append(id)
                values.append(value)
    return IDs, values

def load_pairs(filename):
    pairs = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pairs.add((parts[0], parts[1]))
    return pairs

def cmpt_weights(doc_segs, tokenizer, device, split_method='sbs', weight_scheme='sf'):
    if weight_scheme == 'sf':
        seg_num = len(doc_segs)
        seg_sf = torch.ones(seg_num, dtype=torch.float32).to(device)
        weights = seg_sf/seg_num
    elif weight_scheme == 'sl':
        if split_method == 'sbs':
            seg_lens = [len(tokenizer.tokenize(seg)) for seg in doc_segs]
        elif split_method == 'ofls':
            seg_lens = [len(seg) for seg in doc_segs]
        seg_lens = torch.tensor(seg_lens, dtype=torch.float32).to(device)
        weights = seg_lens/seg_lens.sum()
    return weights

def dedup_docs(A, B, C, device):
    A = [[str(inner) for inner in outer]for outer in A]
    if not (len(A) == len(B) == len(C)):
        raise ValueError("The number of documents in segments, weights, and embeddings is inconsistent")

    A_out, B_out, C_out = [], [], []

    for doc_idx, (seg_list, w_list, e_list) in enumerate(zip(A, B, C)):
        if not (len(seg_list) == len(w_list) == len(e_list)):
            raise ValueError(f"No. {doc_idx} document's segments /weights / embeddings is inconsistent!")

        agg = OrderedDict() 
        for sent, w, emb in zip(seg_list, w_list, e_list):
            if sent in agg:
                agg[sent]["w"] += w
            else:
                agg[sent] = {"w": float(w), "e": emb}

        A_out.append(list(agg.keys()))
        B_out.append([v["w"] for v in agg.values()])
        C_out.append([v["e"] for v in agg.values()])

    return A_out, [torch.tensor(item).to(device) for item in B_out], [torch.tensor(item).to(device) for item in C_out]

'Similarity Score for Search Method'
def cos_sim(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )

def margin_score(embeddings_1, embeddings_2, num_neighbors):

    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    
    cos_score = torch.matmul(normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)).cpu().numpy()
    
    Ix = faiss_cos_search(normalized_embeddings_1.cpu().numpy(), normalized_embeddings_2.cpu().numpy(), num_neighbors)
    Iy = faiss_cos_search(normalized_embeddings_2.cpu().numpy(), normalized_embeddings_1.cpu().numpy(), num_neighbors)
    
    xk_embs = normalized_embeddings_2[Ix, :].view(-1, 768)
    yk_embs = normalized_embeddings_1[Iy, :].view(-1, 768)
    
    repeat_embeddings_1 = normalized_embeddings_1.repeat(1, num_neighbors).view(-1, 768)
    repeat_embeddings_2 = normalized_embeddings_2.repeat(1, num_neighbors).view(-1, 768)
    x_inner = F.cosine_similarity(repeat_embeddings_1, xk_embs).view(-1, num_neighbors).sum(dim=1)/ (2 * num_neighbors)
    y_inner = F.cosine_similarity(repeat_embeddings_2, yk_embs).view(-1, num_neighbors).sum(dim=1)/ (2 * num_neighbors)
    
    NN_score = x_inner.unsqueeze(1) + y_inner
    
    margin_score = cos_score / NN_score.cpu().numpy()

    return margin_score
    
def none_or_str(x):
    if x is None:
        return None
    if isinstance(x, str) and x.lower() == "none":
        return None
    return x
