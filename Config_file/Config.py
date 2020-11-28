import argparse
import yaml

def read_yaml():
    # read in yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path for the config file")
    parser.add_argument("--exp_ids", type=int, nargs='+', default=[0], help="Path for the config file")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="Path for the config file")
    args = parser.parse_args()

    # torch.cuda.set_device(args.gpu)
    with open(args.config) as f:
        opt = yaml.load(f) # 加载出来后是一个字典文件
    return opt, args.gpus, args.exp_ids

read_yaml()