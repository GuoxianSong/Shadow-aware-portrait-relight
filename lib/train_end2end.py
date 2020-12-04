"""
Copyright (C) 2019 Scale Lab.  All rights reserved.
Licensed under the NTU license.
"""

##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
from lib.utils.utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, \
    Timer,write_image2display
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lib.models.end2end_model import Models
from lib.dataset.end2end_dataset import My3DDataset
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/end2end.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume",default='True', action="store_true") #change to True is you need to retrain from pre-train model
    opts = parser.parse_args()

    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)

    # dataset set up
    dataset = My3DDataset(opts=config)
    train_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['nThreads'])


    config['vgg_model_path'] = opts.output_path

    trainer = Models(config)
    trainer.cuda()

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/outputs/logs", model_name))
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config,need_opt=True) if opts.resume else 0

    max_iter = int(config['n_ep']* len(dataset)/config['batch_size'])+1

    while True:

        for it,out_data  in enumerate(train_loader):
            for j in range(len(out_data)):
                out_data[j] = out_data[j].cuda().detach()
            trainer.update_learning_rate()
            with Timer("Elapsed time in update: %f"):
                # Main training code
                if (config['models_name'] == 'end2end'):
                    trainer.gen_update(out_data,config)
                    #trainer.dis_update(Xa_out,X_removal,Xb_out,mask,depth,shadow,light,light_weight,light_label,config)
                    #torch.cuda.synchronize()
            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            if (iterations ) % config['image_display_iter'] == 0:
                write_image2display(iterations, trainer, train_writer)

            # Save network weights
            if (iterations ) % config['snapshot_save_iter']== 0 or iterations+1==max_iter:
                trainer.save(checkpoint_directory, iterations)
            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

if __name__ == '__main__':
    main()