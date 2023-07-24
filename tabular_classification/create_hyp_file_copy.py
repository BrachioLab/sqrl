import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='pre_process_and_train_data.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--seed', type=int, default=10, help='used for resume')

    # parser.add_argument('--hyp_dir', type=str, default="config/hyp/", help='used for resume')
    
    parser.add_argument('--tta_method', type=str, default="rule", help='used for resume')

    args = parser.parse_args()
    return args


def main(args):
    hyp_dir= os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/hyp/") 


    config_file_name = os.path.join(hyp_dir, "hyp_for_test_time_adaptation_" + args.tta_method + ".yaml")

    line_ls = []
    with open(config_file_name) as f:
        for line in f:
            if line.startswith("seed"):
                line = "seed: " + str(args.seed) + "\n"
            line_ls.append(line)

    output_config_file_name = os.path.join(hyp_dir, "hyp_for_test_time_adaptation_" + args.tta_method + "_seed_" + str(args.seed) + ".yaml")

    with open(output_config_file_name, 'w') as f:
        for line in line_ls:
            f.write(line)
        f.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)