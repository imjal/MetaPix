### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class BothOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='results', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        


        # My stuff for launch.py to work
        self.parser.add_argument("--isTrain", type=bool, help="Training Script or Not")
        self.parser.add_argument("--mode", type=str, help="Train, Meta-Train, Finetune")
        self.parser.add_argument("--k", type=int, help="Number of images to finetune on")
        self.parser.add_argument("--T", type=int, help="Number of iterations to finetune on")
        self.parser.add_argument("--train_dataset", type=str, help="Train Dataset specified with all txtfiles")
        self.parser.add_argument("--test_dataset", type=str, help="Test Dataset specified with all txtfiles")
        self.parser.add_argument("--model_checkpoints", type=str, help="Model_checkpoint directory ")
        self.parser.add_argument('--model_list', nargs='+', help='Pass in visualizations that you want to complete')

        self.parser.add_argument('--start_FT_vid', type=int, default= 1, help="Which video from 1-8 that we should start training from")

        self.parser.add_argument('--meta_iter', type=int, help='Pass the number of meta-iterations')
        self.parser.add_argument('--start_meta_iter', type=int, default = 0, help='Pass the number of meta-iterations')
        self.parser.add_argument('--init_weights', type=bool, default=False, help="Determine whether or not to upload pretrained weights for meta-learning")
        self.parser.add_argument('--save_meta_iter', type=int, default=25, help="Determine whether or not to upload pretrained weights for meta-learning")
        self.parser.add_argument('--test_meta_iter', type=int, default=300, help="Determine whether or not to upload pretrained weights for meta-learning")
        self.parser.add_argument('--epsilon', type=float, default=0.1, help = "Meta Learning Rate as detailed in Reptile paper")
        self.parser.add_argument('--only_generator', type=bool, default=False, help="Only train the meta-generator")
        self.parser.add_argument('--one_vid', type=int, default=0, help="Only train the meta-generator")




