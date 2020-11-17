"""Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le
"""
import argparse
import pickle as cp

cmd_opt = argparse.ArgumentParser(description='Argparser for molecule vae')

cmd_opt.add_argument('-saved_model', type=str, default=None, help='saved model')
cmd_opt.add_argument('-save_dir', type=str, default=None, help='save folder')
cmd_opt.add_argument('-ctx', type=str, default='gpu', help='cpu/gpu')

cmd_opt.add_argument('-phase', type=str, default='train', help='train/test')
cmd_opt.add_argument('-batch_size', type=int, default=10, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')

cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp/gcn')
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension of latent layers')
cmd_opt.add_argument('-hidden', type=int, default=0, help='dimension of classification')
cmd_opt.add_argument('-max_lv', type=int, default=1, help='max rounds of message passing')

# target model
cmd_opt.add_argument('-num_epochs', type=int, default=200, help='number of epochs')
cmd_opt.add_argument('-learning_rate', type=float, default=0.01, help='init learning_rate')
cmd_opt.add_argument('-weight_decay', type=float, default=5e-4, help='weight_decay')
cmd_opt.add_argument('-dropout', type=float, default=0.5, help='dropout rate')

# for node classification
cmd_opt.add_argument('-dataset', type=str, default='cora', help='citeseer/cora/pubmed')

# for attack
cmd_opt.add_argument('-num_steps', type=int, default=500000, help='rl training steps')
# cmd_opt.add_argument('-frac_meta', type=float, default=0, help='fraction for meta rl learning')

cmd_opt.add_argument('-meta_test', type=int, default=0, help='for meta rl learning')
cmd_opt.add_argument('-reward_type', type=str, default='binary', help='binary/nll')
cmd_opt.add_argument('-num_mod', type=int, default=1, help='number of modifications allowed')

# for node attack
cmd_opt.add_argument('-bilin_q', type=int, default=1, help='bilinear q or not')
cmd_opt.add_argument('-mlp_hidden', type=int, default=64, help='mlp hidden layer size')
# cmd_opt.add_argument('-n_hops', type=int, default=2, help='attack range')


args, _ = cmd_opt.parse_known_args()
args.save_dir = './results/rl_s2v/{}-gcn'.format(args.dataset)
args.saved_model = 'results/node_classification/{}'.format(args.dataset)
print(args)

def build_kwargs(keys, arg_dict):
    st = ''
    for key in keys:
        st += '%s-%s' % (key, str(arg_dict[key]))
    return st

def save_args(fout, args):
    with open(fout, 'wb') as f:
        cp.dump(args, f, cp.HIGHEST_PROTOCOL)
