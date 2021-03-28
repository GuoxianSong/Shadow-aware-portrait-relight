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
from lib.models.light_model import Models
from lib.dataset.light_dataset import My3DDataset
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
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/light.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume",default='', action="store_true") #change to True is you need to retrain from pre-train model
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

            beauty,label,weight=out_data
            trainer.update_learning_rate()
            with Timer("Elapsed time in update: %f"):
                # Main training code
                if (config['models_name'] == 'light'):
                    trainer.gen_update(beauty,label,weight, config)
                    trainer.dis_update(beauty,label, config)
                    torch.cuda.synchronize()
            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            if (iterations ) % config['image_display_iter'] == 0:
                write_image2display(iterations, trainer, train_writer)
                # image_dis = torchvision.utils.make_grid(trainer.image_display,
                #                                         nrow=trainer.image_display.size(0) // 2)
                # train_writer.add_image('Image', image_dis, iterations)

                #image_dis_ = torchvision.utils.make_grid(trainer.image_input,
                #                                        nrow=trainer.image_input.size(0) // 2)*0.224+0.456
                #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                #train_writer.add_image('Input Image', image_dis_, iterations)

            # Save network weights
            if (iterations+1 ) % config['snapshot_save_iter']== 0 or iterations+1==max_iter:
                trainer.save(checkpoint_directory, iterations)
            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

if __name__ == '__main__':
    main()