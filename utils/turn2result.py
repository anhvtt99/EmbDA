import argparse 
import re
from utils import none_or_str

def main():
    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--data_domain', type=none_or_str, default=None, help='choose from Army, Hiru, ITN, Newsfirst, or None')
    parser.add_argument('--da_method', type=str, default='mean', help='document alignment method,[mean, tkpert, [mean, tkpert]-[sl, sf]-[ot, gmd] (e.g., mean-sf-ot), [mean, tkpert]-bimax]')
    parser.add_argument('--sim_method', type=str, default='cos', help='choose from [cos, margin_score]')
    parser.add_argument('--lang_pair', type=str, default='en-si', help='choose from [en-si, en-ta, si-ta]')
    parser.add_argument('--split_method', type=str, default='ofls', help='choose from [ofls, sbs]')
    parser.add_argument('--overlap', type=float, default= 0.5, help='(The rate of overlap)')
    parser.add_argument('--seg_len', type=int, default=30, help='lenth of segment for sliding window')
    parser.add_argument('--out_path', type=str, default='../da_results', help='path to output')
    args = parser.parse_args()

    output_lines = []

    print("Begining turning output file to result file...")
    lang_pair = args.lang_pair
    da_method = args.da_method
    domain = args.data_domain
    retri = args.sim_method
    if args.split_method == "ofls":
        split_md = f"fl{args.seg_len}_or{args.overlap}"
    else:
        split_md = "sbs"
    
    if domain is None:
        input_file = f"{args.out_path}/{lang_pair}_{split_md}.{retri}.{da_method}.txt"
        output_file = f"{args.out_path}/{lang_pair}_{split_md}.{retri}.{da_method}_result.txt"
    else:
        input_file = f"{args.out_path}/{domain}/{domain}_{lang_pair}_{split_md}.{retri}.{da_method}.txt"
        output_file = f"{args.out_path}/{domain}/{domain}_{lang_pair}_{split_md}.{retri}.{da_method}_result.txt"
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("\t")] 
            if len(parts) >= 3:
                output_lines.append(f"{parts[0]}\t{parts[1]}")
    
    if domain == "ITN":
        output_tmp = []
        for line in output_lines:
            jsons = re.findall(r"\b\d+\.json\b", line)
            if len(jsons) == 2:
                output_tmp.append(f"{jsons[0]}\t{jsons[1]}")
        output_lines = output_tmp

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("Have turned output file to result file!")

if __name__ == '__main__':
    main()
