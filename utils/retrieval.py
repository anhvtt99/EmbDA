import torch
import torch.nn.functional as F
import argparse
import os
import time
import numpy as np
from faiss_search import *
from faiss_search import faiss_L2_search, faiss_cos_search
from transformers import BertTokenizerFast
from utils import *
from da_methods import *

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--embs_path', type=str, default='/work/UTSUROLB/utlb_azam/xwang/20250827_DA/embs', help='path to embeddings')
    parser.add_argument('--out_path', type=str, default='/work/UTSUROLB/utlb_azam/xwang/20250827_DA/da_results', help='path to output')
    parser.add_argument('--data_domain', type=none_or_str, default=None, help='choose from Army, Hiru, ITN, Newsfirst, or None')
    parser.add_argument('--da_method', type=str, default='mean', help='document alignment method, choose from [mean, tkpert, mean-ot, mean-tkpert, mean-maxsim, tkpert-maxsim, mean-bimax, tkpert-bimax]')
    parser.add_argument('--bimax_type', type=str, default='batch', help='calculation type of bimax, choose from [loop, batch]')
    parser.add_argument('--sim_method', type=str, default='cos', help='choose from [cos, margin_score]')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='si')
    parser.add_argument('--split_method', type=str, default='ofls', help='choose from [ofls, sbs]')
    parser.add_argument('--overlap', type=float, default=0.5, help='(The rate of overlap)')
    parser.add_argument('--seg_len', type=int, default=30, help='lenth of segment for sliding window')
    parser.add_argument('--faiss_k', type=int, default=32, help='k nearest neighbors search by faiss')
    parser.add_argument('--faiss_search', type=str, default='cos', help='choose from [L2, cos]')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")

    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    data_domain = args.data_domain
    da_method = args.da_method
    split_method = args.split_method

    # Load data and (Compute weights)
    if split_method == "ofls":
        split_subdir_src = f'{split_method}_fl{args.seg_len}_or{args.overlap}'
        split_subdir_tgt = f'{split_method}_fl{args.seg_len}_or{args.overlap}'
    elif split_method == "sbs":
        split_subdir_src = split_method
        split_subdir_tgt = split_method
    else:
        raise ValueError(f"Unsupported split_method: {split_method}")

    if data_domain is None:
        src_embs_path = os.path.join(args.embs_path, src_lang, split_subdir_src)
        tgt_embs_path = os.path.join(args.embs_path, tgt_lang, split_subdir_tgt)
    else:
        src_embs_path = os.path.join(args.embs_path, data_domain, src_lang, split_subdir_src)
        tgt_embs_path = os.path.join(args.embs_path, data_domain, tgt_lang, split_subdir_tgt)

    src_emb_files_num = count_files(src_embs_path)
    tgt_emb_files_num = count_files(tgt_embs_path)

    print(f"{src_emb_files_num} [{src_lang}] embedding files have been found!")
    print(f"{tgt_emb_files_num} [{tgt_lang}] embedding files have been found!")

    final_dict = {}

    src_IDs, src_values = load_data(src_emb_files_num, src_lang, split_method, src_embs_path)
    tgt_IDs, tgt_values = load_data(tgt_emb_files_num, tgt_lang, split_method, tgt_embs_path)

    if da_method.startswith("mean"):
        src_retrieval_embs = [item["Mean_Emb"] for item in src_values]
        tgt_retrieval_embs = [item["Mean_Emb"] for item in tgt_values]
    elif da_method.startswith("tkpert"):
        src_retrieval_embs = [item["TKPERT_Emb"] for item in src_values]
        tgt_retrieval_embs = [item["TKPERT_Emb"] for item in tgt_values]
    else:
        raise ValueError(f"Unsupported da_method prefix: {da_method}")

    src_retrieval_embs = F.normalize(torch.tensor(src_retrieval_embs).to(device), p=2)
    tgt_retrieval_embs = F.normalize(torch.tensor(tgt_retrieval_embs).to(device), p=2)

    if da_method.endswith(("ot", "gmd")):
        src_seg_embs = [item["Embeds"] for item in src_values]
        tgt_seg_embs = [item["Embeds"] for item in tgt_values]
        src_content = [item["Content"] for item in src_values]
        tgt_content = [item["Content"] for item in tgt_values]

        if "-lidf-" in da_method:
            print("Begin loading sf weights...")
            src_weights = [item["LIDF_weight"] for item in src_values]
            tgt_weights = [item["LIDF_weight"] for item in tgt_values]
            print("Have loaded sf weights!")

        elif ("-sf-" in da_method) or ("-sl-" in da_method):
            if split_method == "sbs":
                src_segs = [sbs_split(doc, src_lang) for doc in src_content]
                tgt_segs = [sbs_split(doc, tgt_lang) for doc in tgt_content]
            elif split_method == "ofls":
                src_segs = [ofls_split(doc, args.seg_len, int(args.seg_len * args.overlap), tokenizer, device)[0] for doc in src_content]
                tgt_segs = [ofls_split(doc, args.seg_len, int(args.seg_len * args.overlap), tokenizer, device)[0] for doc in tgt_content]

            weight_scheme = da_method.split("-")[1]
            print(f"Begin computing {weight_scheme} weights...")
            src_weights = [cmpt_weights(doc_segs, tokenizer, device, split_method=split_method, weight_scheme=weight_scheme) for doc_segs in src_segs]
            tgt_weights = [cmpt_weights(doc_segs, tokenizer, device, split_method=split_method, weight_scheme=weight_scheme) for doc_segs in tgt_segs]
            print(f"Have computed {weight_scheme} weights!")
        else:
            raise ValueError(f"Unsupported weighting scheme in da_method: {da_method}")

        _, src_weights, src_seg_embs = dedup_docs(src_segs, src_weights, src_seg_embs, device)
        _, tgt_weights, tgt_seg_embs = dedup_docs(tgt_segs, tgt_weights, tgt_seg_embs, device)

    elif da_method.endswith("bimax"):
        src_seg_embs = [torch.tensor(item["Embeds"]).to(device) for item in src_values]
        tgt_seg_embs = [torch.tensor(item["Embeds"]).to(device) for item in tgt_values]

    src_retrieval_embs_np = src_retrieval_embs.cpu().detach().numpy()
    tgt_retrieval_embs_np = tgt_retrieval_embs.cpu().detach().numpy()

    faiss_k = min(args.faiss_k, len(tgt_IDs) - 1)
    search_map = {
        "cos": faiss_cos_search,
        "L2": faiss_L2_search,
    }
    I = search_map[args.faiss_search](src_retrieval_embs_np, tgt_retrieval_embs_np, faiss_k)

    result_dict = {}

    print("Begin calculating similarity scores...")
    sim_start_time = time.time()

    if da_method in ("mean", "tkpert"):
        if args.sim_method == 'cos':
            scores = cos_sim(src_retrieval_embs, tgt_retrieval_embs).cpu().detach().numpy()
            for i in range(len(src_IDs)):
                for j in range(faiss_k):
                    result_dict[f"{src_IDs[i]}\t{tgt_IDs[I[i][j]]}"] = scores[i, I[i][j]]

        elif args.sim_method == 'margin':
            if len(src_IDs) >= 20 and len(tgt_IDs) >= 20:
                scores = margin_score(src_retrieval_embs, tgt_retrieval_embs, 4)
            else:
                scores = cos_sim(src_retrieval_embs, tgt_retrieval_embs).cpu().detach().numpy()

            max_indices = np.argsort(-scores, axis=1)[:, :faiss_k]
            for i in range(len(src_IDs)):
                for j in range(faiss_k):
                    result_dict[f"{src_IDs[i]}\t{tgt_IDs[max_indices[i][j]]}"] = scores[i, max_indices[i][j]]

    elif da_method.endswith("ot"):
        for i in range(len(src_IDs)):
            for j in range(faiss_k):
                result_dict[f"{src_IDs[i]}\t{tgt_IDs[I[i][j]]}"] = torch.exp(
                    -cal_ot(
                        src_seg_embs[i], tgt_seg_embs[I[i][j]],
                        src_weights[i].clone(), tgt_weights[I[i][j]].clone(),
                        dist='cos'
                    )
                )

    elif da_method.endswith("gmd"):
        for i in range(len(src_IDs)):
            for j in range(faiss_k):
                result_dict[f"{src_IDs[i]}\t{tgt_IDs[I[i][j]]}"] = torch.exp(
                    -GMD(
                        src_seg_embs[i], tgt_seg_embs[I[i][j]],
                        src_weights[i].clone(), tgt_weights[I[i][j]].clone(),
                        metric="l2",
                        device=device
                    )
                )

    elif da_method.endswith("bimax"):
        if args.bimax_type == "loop":
            for i in range(len(src_IDs)):
                for j in range(faiss_k):
                    result_dict[f"{src_IDs[i]}\t{tgt_IDs[I[i][j]]}"] = bimax_loop(src_seg_embs[i], tgt_seg_embs[I[i][j]])
        elif args.bimax_type == "batch":
            scores = bimax_batch(src_seg_embs, tgt_seg_embs, I, device=device)
            for i in range(len(src_IDs)):
                for j in range(faiss_k):
                    result_dict[f"{src_IDs[i]}\t{tgt_IDs[I[i][j]]}"] = scores[i, j]

    sim_end_time = time.time()
    sim_time = sim_end_time - sim_start_time
    print(f"Finish calculating similarities in {sim_time} seconds!\n")

    sorted_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    used_src = set()
    used_tgt = set()
    for key, value in sorted_dict.items():
        source_num, target_num = key.split('\t')

        if source_num not in used_src and target_num not in used_tgt:
            if da_method.endswith(("ot", "gmd", "bimax")):
                final_dict[key] = value.item()
            else:
                final_dict[key] = value
            used_src.add(source_num)
            used_tgt.add(target_num)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Run all the program in {elapsed_time} sec...\n")

    # Save results
    if data_domain is None:
        output_path = args.out_path
    else:
        output_path = os.path.join(args.out_path, data_domain)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if split_method == "ofls":
        if data_domain is None:
            output_file = os.path.join(
                output_path,
                f'{src_lang}-{tgt_lang}_fl{args.seg_len}_or{args.overlap}.{args.sim_method}.{da_method}.txt'
            )
        else:
            output_file = os.path.join(
                output_path,
                f'{data_domain}_{src_lang}-{tgt_lang}_fl{args.seg_len}_or{args.overlap}.{args.sim_method}.{da_method}.txt'
            )

    elif split_method == "sbs":
        if data_domain is None:
            output_file = os.path.join(
                output_path,
                f'{src_lang}-{tgt_lang}_sbs.{args.sim_method}.{da_method}.txt'
            )
        else:
            output_file = os.path.join(
                output_path,
                f'{data_domain}_{src_lang}-{tgt_lang}_sbs.{args.sim_method}.{da_method}.txt'
            )
    else:
        raise ValueError(f"Unsupported split_method: {split_method}")

    with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
        for key, value in final_dict.items():
            f.write(str(key) + '\t' + str(value) + '\n')

    print("All the work has been done!\n")

if __name__ == '__main__':
    main()