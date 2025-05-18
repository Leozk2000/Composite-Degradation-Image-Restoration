import torch, argparse
from model.OneRestore import OneRestore
from model.Embedder import Embedder

parser = argparse.ArgumentParser()

parser.add_argument("--type", type=str, default = 'OneRestore')
parser.add_argument("--input-file", type=str, default = './ckpts/onerestore_cdd-11.tar')
parser.add_argument("--output-file", type=str, default = './ckpts/onerestore_cdd-11.tar')

args = parser.parse_args()

if args.type == 'OneRestore':
    restorer = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
    restorer_info = torch.load(args.input_file, map_location='cuda:0')
    weights_dict = {}
    for k, v in restorer_info['state_dict'].items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    restorer.load_state_dict(weights_dict)
    torch.save(restorer.state_dict(), args.output_file)
elif args.type == 'Embedder':
    combine_type = combine_type=[
    'blur',          # 0
    'blur_haze',     # 1
    'clear',         # 2
    'haze',          # 3
    'haze_rain',     # 4
    'haze_snow',     # 5
    'low',           # 6
    'low_haze',      # 7
    'low_haze_rain', # 8
    'low_haze_snow', # 9
    'low_rain',      # 10
    'low_snow',      # 11
    'rain',          # 12
    'snow'           # 13
]
    embedder = Embedder(combine_type).to("cuda" if torch.cuda.is_available() else "cpu")
    embedder_info = torch.load(args.input_file)
    embedder.load_state_dict(embedder_info['state_dict'])
    torch.save(embedder.state_dict(), args.output_file)
else:
    print('ERROR!')

