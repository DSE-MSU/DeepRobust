import torch
import numpy as np
import argparse
import attack

def run_attack(method, model, **kwargs):
    pass

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run attack algorithms.")

    parser.add_argument("--attack_method", 
                        type = string, 
                        default = 'PGD_attack', 
                        help = "choose a attack algorithm")
    parser.add_argument("--epsilon", type = double, default = 0.3)
    parser.add_argument("--num_steps", type = int, default = 40)
    parser.add_argument("--step_size", type = double, default = 0.01)


if __name__ == "__main__":
    # read arguments
    args = parameter_parser() # read argument and creat an argparse object
    tab_printer(args)
    model = Net()
    print("Load network.")
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    model.eval()

    if(args.attack_method = "PGD_attack"):
        attack_method = attack.pgd.PGD(model)
        run_attack(attack_method, args.epsilon, args.num_steps, args.step_size)

