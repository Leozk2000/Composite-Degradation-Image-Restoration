import os, time, torch, argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import numpy as np
from torchvision import transforms
from makedataset import Dataset
from utils.utils import print_args, load_restore_ckpt_with_optim, load_embedder_ckpt, adjust_learning_rate, data_process, tensor_metric, load_excel, save_checkpoint
from model.loss import Total_loss
import shutil

from PIL import Image

transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('> Model Initialization...')

    embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=args.embedder_model_path)
    restorer, optimizer, cur_epoch = load_restore_ckpt_with_optim(device, freeze_model=False, ckpt_name=args.restore_model_path, lr=args.lr)
    loss = Total_loss(args)
    
    print('> Loading dataset...')
    data = Dataset(args.train_input)
    dataset = DataLoader(dataset=data, num_workers=args.num_works, batch_size=args.bs, shuffle=True)
    
    print('> Start training...')
    print(f'Training from epoch {cur_epoch} to {args.epoch}')
    start_all = time.time()
    train(restorer, embedder, optimizer, loss, cur_epoch, args, dataset, device)
    end_all = time.time()
    print('Whole Training Time:' +str(end_all-start_all)+'s.')


def train(restorer, embedder, optimizer, loss, cur_epoch, args, dataset, device):
    metric = []
    best_psnr = 0
    log_path = os.path.join(args.output, "training_log.txt") 
    with open(log_path, "a") as log_file:  # "w" mode clears previous logs
        log_file.write("Training Log for OneRestore\n")
        log_file.write("=" * 50 + "\n")
    # print_freq = 200  # Print frequency
    
    class_names = ['blur', 'blur_haze', 'clear', 'haze', 'haze_rain', 
                  'haze_snow', 'low', 'low_haze', 'low_haze_rain', 
                  'low_haze_snow', 'low_rain', 'low_snow', 'rain', 'snow']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    total_dataset_stats = {name: 0 for name in class_names}

    for epoch in range(cur_epoch, args.epoch):
        optimizer = adjust_learning_rate(optimizer, epoch, args.adjust_lr)
        learnrate = optimizer.param_groups[-1]['lr']
        restorer.train()
        epoch_metrics = {i: {'count': 0, 'psnr': 0, 'ssim': 0, 'loss': 0} for i in range(14)}
        epoch_total_loss = 0
        epoch_batches = 0
        epoch_loss_values = []

        print(f"\nEpoch: [{epoch+1}/{args.epoch}] Learning rate: {learnrate:.9f}")
        
        for i, data in enumerate(dataset, 0):
            pos, inp, neg = data_process(data, args, device)
            
            # Update dataset stats
            for degra_type in inp[1]:
                total_dataset_stats[degra_type] += 1

            # Forward pass
            text_embedding, _, _ = embedder(inp[1], 'text_encoder')
            out = restorer(inp[0], text_embedding)
            # out = torch.clamp(out, 0, 1)  # Ensure output is in [0,1] range

            # Backward pass
            restorer.zero_grad()
            total_loss = loss(inp, pos, neg, out)
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(restorer.parameters(), max_norm=1.0) #gradient clipping
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_batches += 1    
            epoch_loss_values.append(total_loss.item())

            # Calculate metrics
            psnr = tensor_metric(pos, out, 'PSNR', data_range=1)
            ssim = tensor_metric(pos, out, 'SSIM', data_range=1)

            # Update class-specific metrics
            for idx, degra_type in enumerate(inp[1]):
                class_idx = class_to_idx[degra_type]
                curr_psnr = tensor_metric(pos[idx:idx+1], out[idx:idx+1], 'PSNR', data_range=1)
                curr_ssim = tensor_metric(pos[idx:idx+1], out[idx:idx+1], 'SSIM', data_range=1)
                
                epoch_metrics[class_idx]['count'] += 1
                epoch_metrics[class_idx]['psnr'] += curr_psnr
                epoch_metrics[class_idx]['ssim'] += curr_ssim
                epoch_metrics[class_idx]['loss'] += total_loss.item()

            print(f"Batch [{i+1}/{len(dataset)}] Loss: {total_loss.item():.4f} PSNR: {psnr:.4f} SSIM: {ssim:.4f}")

        # Log epoch-level metrics
        avg_epoch_loss = epoch_total_loss / epoch_batches

        with open(log_path, "a") as log_file:  # Open in append mode
            log_file.write(f"\nEpoch {epoch+1} Summary:\n")
            log_file.write(f"{'Class':<15} {'Count':>5} {'PSNR':>8} {'SSIM':>8} {'Loss':>8}\n")
            log_file.write("-" * 50 + "\n")

            for class_idx in range(14):
                if epoch_metrics[class_idx]['count'] > 0:
                    avg_psnr = epoch_metrics[class_idx]['psnr'] / epoch_metrics[class_idx]['count']
                    avg_ssim = epoch_metrics[class_idx]['ssim'] / epoch_metrics[class_idx]['count']
                    avg_loss = epoch_metrics[class_idx]['loss'] / epoch_metrics[class_idx]['count']


                    log_entry = (f"{class_names[class_idx]:<15} {epoch_metrics[class_idx]['count']:>5d} "
                                f"{avg_psnr:>8.2f} {avg_ssim:>8.4f} {avg_loss:>8.4f}\n")

                    print(log_entry, end="")  # Print to console
                    log_file.write(log_entry)  # Write to log file

        # Test and checkpoint
        psnr_t1, ssim_t1, psnr_t2, ssim_t2 = test(args, restorer, embedder, device, epoch)

        
        metric.append([psnr_t1, ssim_t1, psnr_t2, ssim_t2])
        test_log = f"\nTest Results - PSNR: {psnr_t1:.4f} SSIM: {ssim_t1:.4f}\n"
        print(test_log)  # Print to console
        with open(log_path, "a") as log_file:
            log_file.write(test_log)  # Write to log file

        # Save best model
        if psnr_t1 > best_psnr:
            best_psnr = psnr_t1
            save_checkpoint(
                {'epoch': epoch + 1,
                 'state_dict': restorer.state_dict(),
                 'optimizer': optimizer.state_dict()},
                args.save_model_path, epoch+1, psnr_t1, ssim_t1, psnr_t2, ssim_t2
            )
        else:
            save_checkpoint(
                {'epoch': epoch + 1,
                 'state_dict': restorer.state_dict(),
                 'optimizer': optimizer.state_dict()},
                args.save_model_path, epoch+1, psnr_t1, ssim_t1, psnr_t2, ssim_t2
            )
        load_excel(metric)    


def test(args, restorer, embedder, device, epoch=-1):
    log_path = os.path.join(args.output, "training_log.txt")
    combine_type = args.degr_type
    class_metrics = {deg_type: {
        'psnr_text': 0,
        'ssim_text': 0,
        'psnr_img': 0,
        'ssim_img': 0,
        'count': 0
    } for deg_type in combine_type}
    # psnr_1, psnr_2, ssim_1, ssim_2 = 0, 0, 0, 0
    os.makedirs(args.output,exist_ok=True)

    # Use clear images (index 2) as reference
    clear_type = combine_type[2]  # 'clear'
    
    # Skip clear type in degradation loop
    test_types = [t for i, t in enumerate(combine_type) if i != 2]
    
    for degradation_type in test_types:
        file_list = os.listdir(f'{args.test_input}/{degradation_type}/')
        print(f"\nProcessing {degradation_type} - {len(file_list)} images")
        for j in range(len(file_list)):
            # Load clear image as reference
            hq = Image.open(f'{args.test_input}/{clear_type}/{file_list[j]}')
            # Load degraded image
            lq = Image.open(f'{args.test_input}/{degradation_type}/{file_list[j]}')
            
            restorer.eval()
            with torch.no_grad():
                # Rest of the processing...
                lq_re = torch.Tensor((np.array(lq)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                lq_em = transform_resize(lq).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                hq = torch.Tensor((np.array(hq)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

                starttime = time.time()

                text_embedding_1,_,text_1 = embedder([degradation_type],'text_encoder')
                text_embedding_2,_, text_2 = embedder(lq_em,'image_encoder')
                out_1 = restorer(lq_re, text_embedding_1)
                if text_1 != text_2:
                    print(text_1, text_2)
                    out_2 = restorer(lq_re, text_embedding_2)
                else:
                    out_2 = out_1

                # Calculate metrics
                psnr_text = tensor_metric(hq, out_1, 'PSNR', data_range=1)
                ssim_text = tensor_metric(hq, out_1, 'SSIM', data_range=1)
                psnr_img = tensor_metric(hq, out_2, 'PSNR', data_range=1)
                ssim_img = tensor_metric(hq, out_2, 'SSIM', data_range=1)

                # Update class metrics
                class_metrics[degradation_type]['psnr_text'] += psnr_text
                class_metrics[degradation_type]['ssim_text'] += ssim_text
                class_metrics[degradation_type]['psnr_img'] += psnr_img
                class_metrics[degradation_type]['ssim_img'] += ssim_img
                class_metrics[degradation_type]['count'] += 1

                if epoch % 50 == 0:  # Save every 50
                    imwrite(torch.cat((lq_re, out_1, out_2, hq), dim=3), 
                        f"{args.output}/{file_list[j][:-4]}_{epoch}_{degradation_type}.png", 
                        range=(0, 1))
                  
                # # Calculate metrics using clear image as reference
                # psnr_1 += tensor_metric(hq, out_1, 'PSNR', data_range=1)
                # ssim_1 += tensor_metric(hq, out_1, 'SSIM', data_range=1)
                # psnr_2 += tensor_metric(hq, out_2, 'PSNR', data_range=1)
                # ssim_2 += tensor_metric(hq, out_2, 'SSIM', data_range=1)

    # Calculate averages and log results
    total_psnr_text = 0
    total_ssim_text = 0
    total_psnr_img = 0
    total_ssim_img = 0
    total_count = 0
    
    print("\n" + "="*80)
    print("DETAILED TEST RESULTS:")
    print("="*80)
    
    with open(log_path, "a") as log_file:
        if epoch == -1:
            header = "\nPretrained Model Test Results:"
        else:
            header = f"\nEpoch {epoch+1} Test Results:"
        
        log_file.write(header + "\n")
        print(header)
        
        log_file.write("=" * 80 + "\n")
        header = f"{'Degradation Type':<15} {'Count':>6} {'PSNR (Text)':>12} {'SSIM (Text)':>12} {'PSNR (Img)':>12} {'SSIM (Img)':>12}"
        log_file.write(header + "\n")
        log_file.write("-" * 80 + "\n")
        
        print("=" * 80)
        print(header)
        print("-" * 80)
        
        for deg_type in test_types:
            metrics = class_metrics[deg_type]
            count = metrics['count']
            if count > 0:
                avg_psnr_text = metrics['psnr_text'] / count
                avg_ssim_text = metrics['ssim_text'] / count
                avg_psnr_img = metrics['psnr_img'] / count
                avg_ssim_img = metrics['ssim_img'] / count
                
                result_line = f"{deg_type:<15} {count:>6d} {avg_psnr_text:>12.4f} {avg_ssim_text:>12.4f} {avg_psnr_img:>12.4f} {avg_ssim_img:>12.4f}"
                log_file.write(result_line + "\n")
                print(result_line)
                
                total_psnr_text += metrics['psnr_text']
                total_ssim_text += metrics['ssim_text']
                total_psnr_img += metrics['psnr_img']
                total_ssim_img += metrics['ssim_img']
                total_count += count
        
        log_file.write("-" * 80 + "\n")
        avg_psnr_text = total_psnr_text / total_count
        avg_ssim_text = total_ssim_text / total_count
        avg_psnr_img = total_psnr_img / total_count
        avg_ssim_img = total_ssim_img / total_count
        
        overall_line = f"{'Overall':<15} {total_count:>6d} {avg_psnr_text:>12.4f} {avg_ssim_text:>12.4f} {avg_psnr_img:>12.4f} {avg_ssim_img:>12.4f}"
        log_file.write(overall_line + "\n")
        log_file.write("=" * 80 + "\n\n")
        
        print("-" * 80)
        print(overall_line)
        print("=" * 80)

        # Original format summary print
        print(f"\nTest Results - PSNR: {avg_psnr_text:.4f} SSIM: {avg_ssim_text:.4f}")
    
    return avg_psnr_text, avg_ssim_text, avg_psnr_img, avg_ssim_img

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "OneRestore Training")

    # load model
    parser.add_argument("--embedder-model-path", type=str, default = "./ckpts/embedder_model.tar", help = 'embedder model path')
    parser.add_argument("--restore-model-path", type=str, default = "./ckpts/test/OneRestore_model_114_27.3705_0.8285_27.0961_0.8247.tar", help = 'restore model path')
    parser.add_argument("--save-model-path", type=str, default = "./ckpts/test/", help = 'restore model path')

    parser.add_argument("--epoch", type=int, default = 300, help = 'epoch number')
    parser.add_argument("--bs", type=int, default = 4, help = 'batchsize')
    parser.add_argument("--lr", type=float, default = 2e-4, help = 'learning rate')
    parser.add_argument("--adjust-lr", type=int, default = 20, help = 'adjust learning rate')
    parser.add_argument("--num-works", type=int, default = 4, help = 'number works')
    # Loss weights
    parser.add_argument("--loss-weight", type=tuple, 
                        default = (0.6,     # smooth_l1 weight
                                   0.3,     # mssim weight
                                   0.1,    # contrast weight
                                   0),   # charbonnier weight
                        help = 'loss weights')
    parser.add_argument("--degr-type", type=list, default=[
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
    ], help = 'degradation type')    
    parser.add_argument("--train-input", type=str, default = "./dataset.h5", help = 'train data')
    parser.add_argument("--test-input", type=str, default = "./data/CDD-11_test", help = 'test path')
    parser.add_argument("--output", type=str, default = "./result/", help = 'output path')

    argspar = parser.parse_args()

    print_args(argspar)
    main(argspar)
