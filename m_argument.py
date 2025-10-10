import argparse

def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_mode", type=str, default="AID_UCMD_NWPU")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--training_phase", type=str, default="phase1", choices=["phase1", "phase2"],
                    help="training_phase: phase1 or phase2")
    parser.add_argument("--load_model", type=str, default="",
                    help="phase2 load_model")
    parser.add_argument('--lr', type=float, default=5e-5, help='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--temperature', type=float, default=1, help='temperature')
    parser.add_argument('--lambda_ce', type=float, default=1, help='ce')
    parser.add_argument('--lambda_wce', type=float, default=1, help='wce')
    parser.add_argument('--lambda_OSD', type=float, default=1, help='OSD')
    parser.add_argument('--lambda_consistency', type=float, default=0.1, help='kl')
    parser.add_argument("--phase", type=str, default="train_uda", choices=["train_uda", "test"], help='train_uda or test')
    parser.add_argument("--log_dir", type=str, default="logs", help='logs')

    args = parser.parse_args()

    # data
    if args.dataset_mode in ["AID_UCMD_NWPU","NWPU_AID_UCMD","UCMD_NWPU_AID"]:
        args.class_list = [
            'agricultural',           # 0
            'baseball diamond',       # 1
            'beach',                  # 2
            'dense residential',      # 3
            'forest',                 # 4
            'medium residential',     # 5
            'parking lot',            # 6
        ]
        args.source_classes = [0, 1, 2, 3, 4, 5, 6]  

        args.class_list.append('unknown')
        args.class_num = len(args.class_list)
    else:
        raise ValueError("Invalid dataset mode!")

    args.unknown_class_index = args.class_num - 1

    return args