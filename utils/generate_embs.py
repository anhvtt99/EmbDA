import json
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F
import torch
import os
import argparse
import json
import glob
from utils import*

def main():
    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--overlap', type=float, default= 0.5, help='(The rate of overlap)')
    parser.add_argument('--seg_len', type=int, default=30, help='lenth of segment for sliding window')
    parser.add_argument('--split_method', type=str, default='ofls', help='choose from [sbs (sentence-based segmentation), ofls (overlapping fixed-length segmentation)]')
    parser.add_argument('--chunk_size', type=int, default=2048, help='chunk size used to split document segments into batches for embedding')
    parser.add_argument('--lang', type=str, default='en', help='choose from en, si, ta')
    parser.add_argument('--data_domain', type=none_or_str, default=None, help='choose from Army, Hiru, ITN, Newsfirst, or None')
    parser.add_argument('--data_path', type=str, default='../fernando_data', help='Path to data')
    parser.add_argument('--out_path', type=str, default='../embs', help='Path to output file')
    parser.add_argument('--J', type=int, default= 16, help='The numbers of PERT winodws')
    parser.add_argument('--shape', type=int, default= 16, help='Controls the weight of the most likely value in the determination of the mean for PERT distribution')
    parser.add_argument('--save_num', type=int, default= 500, help='Number of docs to save')
    parser.add_argument('--input_format', type=str, default='json', choices=['json', 'txt'],
                        help='Input file format: json or txt')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    'Load data'
    print(f"Begin reading the [{args.lang}] data...")
    
    if args.data_domain is None:
        docs_path = os.path.join(args.data_path, args.lang)
    else:
        docs_path = os.path.join(args.data_path, args.data_domain, args.lang)
    
    data_dict = {}
    
    if args.input_format == "json":
        input_files = glob.glob(os.path.join(docs_path, "*.json"))
        for file in input_files:
            with open(file, "r", encoding="utf-8") as f:
                data_dict[os.path.basename(file)] = json.load(f)
    
    elif args.input_format == "txt":
        input_files = glob.glob(os.path.join(docs_path, "*.txt"))
        for file in input_files:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            data_dict[os.path.basename(file)] = {"Content": text}
    
    doc_num = len(data_dict)
    print(f"{doc_num} [{args.lang}] docs have been read...")
    
    del input_files

    'load model'
    print("Use LaBSE model for embedding. Loading model...")
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    model = BertModel.from_pretrained("setu4993/LaBSE").to(device)
    model = model.eval()
    print("LaBSE model has been loaded!")


    'make output folds'
    if args.split_method == "sbs":
        if args.data_domain is None:
            out_file_path = os.path.join(args.out_path, args.lang, "sbs")
        else:
            out_file_path = os.path.join(args.out_path, args.data_domain, args.lang, "sbs")
    
    elif args.split_method == "ofls":
        segment_len = args.seg_len
        overlap_rate = args.overlap
        overlap = int(segment_len * overlap_rate)
    
        if args.data_domain is None:
            out_file_path = os.path.join(
                args.out_path, args.lang, f"ofls_fl{args.seg_len}_or{args.overlap}"
            )
        else:
            out_file_path = os.path.join(
                args.out_path, args.data_domain, args.lang, f"ofls_fl{args.seg_len}_or{args.overlap}"
            )

    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)  


    'calculate LIDF weights'
    segments_list = []
    with torch.no_grad():
        print(f"Begin calculating LIDF weights with [{args.lang}] documents ...")
        
        for _, document in data_dict.items():
            content = document["Content"]
            if args.split_method == "sbs":
                doc_segments = sbs_split(content, args.lang)
            elif args.split_method == "ofls":
                doc_segments, _ = ofls_split(content, segment_len, overlap, tokenizer, device)
            segments_list.append(doc_segments)
        LIDF_weight_list = cal_LIDF(segments_list)
        
        index=0
        for _, document in data_dict.items():
            document["LIDF_weight"] = LIDF_weight_list[index]
            index += 1
            
        print(f"Done calculation for LIDF weights with [{args.lang}] document!")
            
        del segments_list


    'create embeddings'
    print(f"Begin embedding with [{args.lang}] documents ...")

    count= 0
    file_count = 0
    save_num = args.save_num
    chunk_size = args.chunk_size
    output_dict = {}
    output_dict['lr_docs'] = []
    with torch.no_grad():
        for file_name, document in data_dict.items():
            dict_tmp = {}
            content = document["Content"]
            doc_tkn = tokenizer.tokenize(content)
            if args.split_method == "sbs":
                doc_segments = sbs_split(content, args.lang)
                batches = split_list(doc_segments, chunk_size)
            elif args.split_method == "ofls":
                doc_segments, batches = ofls_split(content, segment_len, overlap, tokenizer, device, chunk_size)
            doc_embs = []
            for item in batches:
                if args.split_method == "sbs": 
                    inputs = tokenizer(item, return_tensors="pt", padding=True, truncation=True).to(device)
                elif args.split_method == "ofls":
                    inputs = item
                outputs = model(**inputs)
                doc_embs_tmp = outputs.pooler_output
                doc_embs.append(doc_embs_tmp)
            doc_embs = torch.cat(doc_embs, 0)
            'normalize sentence embeddings'
            doc_embs = F.normalize(doc_embs, p=2)
            'mean-pool vector'
            mean_emb = doc_embs.mean(dim=0)
            'tkpert vector'
            pert_weights = pert_windows(args.J, len(doc_segments), args.shape).to(device)
            LIDF_wight = torch.tensor(document["LIDF_weight"]).to(device, dtype=torch.float32)
            doc_embs_with_LIDF = doc_embs * LIDF_wight.view(len(LIDF_wight), 1)
            V_embs = F.normalize(torch.mm(pert_weights, doc_embs_with_LIDF), p=2)
            flattened_V = V_embs.reshape(-1)
            
            'add features into dict'
            document['Embeds'] = doc_embs.cpu().numpy().tolist()
            document['Mean_Emb'] = mean_emb.cpu().numpy().tolist()
            document['TKPERT_Emb'] = flattened_V.cpu().numpy().tolist()
            document['LIDF_weight'] = document['LIDF_weight'].tolist()
            document['num_tokens'] = len(doc_tkn)

            count += 1
            dict_tmp[file_name] = document
            output_dict['lr_docs'].append(dict_tmp)
            
            if count % 100 == 0:
                print(f'Process [{args.lang}] docs in {count}/{doc_num}')
            if count % save_num == 0:
                print(f"Write the No.{file_count} [{args.lang}] embs into json file...")
                with open(f'{out_file_path}/{file_count}-emb.json', 'w') as json_file:
                    json.dump(output_dict, json_file, ensure_ascii=False, separators=(',', ': '))
                file_count += 1
                output_dict = {}
                output_dict['lr_docs'] = []
            if count >= doc_num:
                print(f"Write the No.{file_count} [{args.lang}] embs into json file...")
                with open(f'{out_file_path}/{file_count}-emb.json', 'w') as json_file:
                    json.dump(output_dict, json_file, ensure_ascii=False, separators=(',', ': '))
            
    print(f"Done the work for [{args.lang}] data!")

if __name__ == '__main__':
    main()