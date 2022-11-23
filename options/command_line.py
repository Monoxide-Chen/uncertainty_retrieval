import argparse
import sys

parser = argparse.ArgumentParser(description='Options for CoSMo.pytorch')

#########################
# Load Template
#########################
parser.add_argument('--config_path', type=str, default='', help='config json path')

#########################
# Trainer Settings
#########################
parser.add_argument('--trainer', type=str, default="tirg", help='Select Trainer')
parser.add_argument('--epoch', type=int, default=80, help='epoch (default: 80)')
parser.add_argument('--evaluator', type=str, default="simple", help='Select Evaluator')

#########################
# Language Template
#########################
parser.add_argument('--vocab_path', type=str, default='', help='Vocabulary path')
parser.add_argument('--vocab_threshold', type=int, default=0, help='Vocabulary word count threshold')

#########################
# Dataset / DataLoader Settings
#########################
parser.add_argument('--dataset', type=str, default='fashionIQ_dress', help='Dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--num_workers', type=int, default=16, help='The Number of Workers')
parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle Dataset')

#########################
# Image Transform Settings
#########################
parser.add_argument('--use_transform', type=bool, default=True, help='Use Transform')
parser.add_argument('--img_size', type=int, default=224, help='Img Size')

#########################
# Loss Settings
#########################
parser.add_argument('--metric_loss', type=str, default="batch_based_classification_loss", help='Metric Loss Code')
parser.add_argument('--gamma_scale', type=float, default=1, help='change the balance of two loss functions')
#########################
# Encoder Settings
#########################
parser.add_argument('--feature_size', type=int, default=512, help='Image Feature Size')
parser.add_argument('--text_feature_size', type=int, default=512, help='Text Feature Size')
parser.add_argument('--word_embedding_size', type=int, default=512, help='Word Embedding Size')
parser.add_argument('--image_encoder', type=str, default='resnet50_layer4', help='Image Model')
parser.add_argument('--stride', type=bool, default=True, help='ResNet50 stride')
parser.add_argument('--text_encoder', type=str, default="lstm", help='Text Model')

#########################
# Composition Model Settings
#########################
parser.add_argument('--compositor', type=str, default="transformer", help='Composition Model')
parser.add_argument('--norm_scale', type=float, default=4, help='Norm Scale')
parser.add_argument('--num_heads', type=int, default=8, help='Num Heads')
parser.add_argument('--global_styler', type=str, default='global2', help='Global Styler')

#########################
# Augmenter Model Settings
#########################
parser.add_argument('--augmenter', type=str, default="normal_gaussian", help='Normal Augmenter or None')
parser.add_argument('--alpha_scale', type=float, default=1, help='change the scale of alpha noise')
parser.add_argument('--beta_scale', type=float, default=1, help='change the scale of beta noise')

#########################
# Optimizer Settings
#########################
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RAdam'], help='Optimizer')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate (default: 2e-3)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='l2 regularization lambda (default: 5e-5)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--warmup_iters', type=int, default=5, help='num epochs for warmup learning rate')
parser.add_argument('--decay_step', type=int, default=35, help='num epochs for first decaying learning rate')
parser.add_argument('--decay_step_second', type=int, default=45, help='num epochs for second decaying learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay gamma')
parser.add_argument('--lr_scheduler', type=str, default='MultiStepWithWarmup', help='learning rate scheduler')

#########################
# Logging Settings
#########################
parser.add_argument('--topk', type=str, default='1,5,10,50', help='topK recall for evaluation')
parser.add_argument('--wandb_project_name', type=str, default='UR', help='Weights & Biases project name')
parser.add_argument('--wandb_account_name', type=str, default='your_account_name', help='Weights & Biases account name')

#########################
# Resume Training
#########################
parser.add_argument('--checkpoint_path', type=str, default='', help='Path to saved checkpoint file')

#########################
# Misc
#########################
parser.add_argument('--device_idx', type=str, default='0,1', help='Gpu idx')
parser.add_argument('--random_seed', type=int, default=-1, help=
                    'Random seed value is greater than or equal to -1, which is equal to -1 getting random value from non-negative integer')
parser.add_argument('--experiment_dir', type=str, default='experiments', help='Experiment save directory')
parser.add_argument('--experiment_description', type=str, default='NO', help='Experiment description')


def _get_user_defined_arguments(argvs):
    prefix, conjugator = '--', '='
    return [argv.replace(prefix, '').split(conjugator)[0] for argv in argvs]


def load_config_from_command():
    user_defined_argument = _get_user_defined_arguments(sys.argv[1:])

    configs = vars(parser.parse_args())
    user_defined_configs = {k: v for k, v in configs.items() if k in user_defined_argument}
    return configs, user_defined_configs
