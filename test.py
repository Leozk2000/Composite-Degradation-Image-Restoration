import os, time, argparse
from PIL import Image
import numpy as np


import torch
from torchvision import transforms

from torchvision.utils import save_image as imwrite
from utils.utils import print_args, load_restore_ckpt, load_embedder_ckpt

transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

def process_image(lq, max_size=2048):
    # Get original size
    w, h = lq.size
    should_resize = max(w, h) > max_size
    scale = max_size / max(w, h) if should_resize else 1
    
    if should_resize:
        new_w = int(w * scale)
        new_h = int(h * scale)
        lq = lq.resize((new_w, new_h), Image.LANCZOS)
    
    return lq

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #train
    print('> Model Initialization...')

    embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=args.embedder_model_path)
    restorer = load_restore_ckpt(device, freeze_model=True, ckpt_name=args.restore_model_path)

    os.makedirs(args.output,exist_ok=True)
    
    files = os.listdir(argspar.input)
    time_record = []
    for i in files:
        # Clear GPU cache before each image
        torch.cuda.empty_cache()

        lq = Image.open(f'{argspar.input}/{i}')
        # Check and resize if image is too large
        lq = process_image(lq)

        with torch.no_grad():
            lq_re = torch.Tensor((np.array(lq)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            lq_em = transform_resize(lq).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

            start_time = time.time()
            
            if args.prompt == None:
                text_embedding, _, [text] = embedder(lq_em,'image_encoder')
                print(f'This is {text} degradation estimated by visual embedder.')
            else:
                text_embedding, _, [text] = embedder([args.prompt],'text_encoder')
                print(f'This is {text} degradation generated by input text.')
            
            out = restorer(lq_re, text_embedding)

            run_time = time.time()-start_time
            time_record.append(run_time)

            if args.concat:
                out = torch.cat((lq_re, out), dim=3)

            imwrite(out, f'{args.output}/{i}', range=(0, 1))

            print(f'{i} Running Time: {run_time:.4f}.')
    print(f'Average time is {np.mean(np.array(run_time))}')
            

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "OneRestore Running")

    # load model
    # parser.add_argument("--embedder-model-path", type=str, default = "./ckpts/embedder_model.tar", help = 'embedder model path')
    # parser.add_argument("--restore-model-path", type=str, default = "./ckpts/onerestore_model.tar", help = 'restore model path')
    parser.add_argument("--embedder-model-path", type=str, default = "./ckpts(newblur)/embedder_model.tar", help = 'embedder model path')
    parser.add_argument("--restore-model-path", type=str, default = "./ckpts(newblur)/onerestore_model.tar", help = 'restore model path')

    # select model automatic (prompt=False) or manual (prompt=True, text={'clear', 'low', 'haze', 'rain', 'snow',\
    #                'low_haze', 'low_rain', 'low_snow', 'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow'})
    parser.add_argument("--prompt", type=str, default = None, help = 'prompt')

    parser.add_argument("--input", type=str, default = "./in3", help = 'image path')
    # parser.add_argument("--input", type=str, default = r"C:\Users\Admin\Desktop\fyp2\OneRestore\data\CDD-11_test\blur_haze", help = 'image path')
    parser.add_argument("--output", type=str, default = "./out3", help = 'output path')
    # parser.add_argument("--output", type=str, default = r"C:\Users\Admin\Desktop\fyp2\OneRestore\benchresult\RainDS13", help = 'output path')
    parser.add_argument("--concat", action='store_true', help = 'output path')

    argspar = parser.parse_args()

    print_args(argspar)

    main(argspar)
