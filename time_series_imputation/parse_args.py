import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    # parser.add_argument('--model', type=str, default='DHMM_cluster', help='model name')
    # parser.add_argument('--dataset', type=str, default='climate_NY', help='name of dataset')
#     parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
    # parser.add_argument('--input_dir', type=str, help="std of the initial phi table")
    # parser.add_argument('--alpha', type=float, default=10, help="std of the initial phi table")
    # parser.add_argument('--train_rate', type=float, default=0.8, help="std of the initial phi table")
    # parser.add_argument('--p_hint', type=float, default=0.9, help="std of the initial phi table")
    parser.add_argument('--missing_ratio', type=float, default=0.2, help="std of the initial phi table")
    parser.add_argument('--input', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--output', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--log_path', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--cache_path', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--task_name', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--dataset', type=str, default="physionet", choices=["physionet", "mimic3"], help="dataset name")
    parser.add_argument('--model_type', type=str, default="mTan", choices=["mTan", "csdi", "saits"], help="std of the initial phi table")
    
    
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])

    parser.add_argument('--imputed', action='store_true', help='specifies what features to extract')
    parser.add_argument('--do_train', action='store_true', help='specifies what features to extract')
    parser.add_argument('--classify_task', action='store_true', help='specifies what features to extract')
    parser.add_argument('--tta_method', type=str, default="rule", help='used for resume')

    # parser.add_argument('--epochs', type=int, default=10, help="std of the initial phi table")
    # parser.add_argument('--batch_size', type=int, default=16, help="std of the initial phi table")
    # parser.add_argument('--lr', type=float, default=0.001, help="std of the initial phi table")
    # parser.add_argument('--alpha', type=float, default=100, help="std of the initial phi table")
    # parser.add_argument('--beta', type=float, default=2, help="std of the initial phi table")
    # parser.add_argument('--kl', action='store_true')
    # parser.add_argument('--classify-pertp', action='store_true')

    args = parser.parse_args()

    return args


