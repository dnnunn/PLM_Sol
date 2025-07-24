import argparse
import yaml
import torch
from models import *  # For loading classes specified in config
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_dataset import EmbeddingsDataset
from datasets.transforms import *
from solver import Solver
from utils.general import padded_permuted_collate, seed_all


def evaluate(args):
    """Loads a pre-trained model and evaluates it on a given test set."""
    seed_all(args.seed)

    # 1. Load the test dataset using paths from the config
    print(f"\n📖 Loading test set from config paths:")
    print(f"   - Embeddings: {args.test_embeddings}")
    print(f"   - Remapping: {args.test_remapping}")
    transform = transforms.Compose([SolubilityToInt(), ToTensor()])
    test_set = EmbeddingsDataset(args.test_embeddings, args.test_remapping, args.unknown_solubility,
                                 key_format=args.key_format, max_length=args.max_length,
                                 embedding_mode=args.embedding_mode, transform=transform)

    print(f"\n✅ Loaded test set with {len(test_set)} samples.")

    # 2. Initialize the model from the same architecture
    model = globals()[args.model_type](embeddings_dim=test_set[0][0].shape[-1], **args.model_parameters)
    print(f"🧠 Model '{args.model_type}' initialized.")

    # 3. Load the pre-trained baseline checkpoint
    try:
        # The checkpoint path is now passed via command line and not from the config
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"💾 Successfully loaded baseline model weights from: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # 4. Set up the solver and run evaluation
    solver = Solver(model, args, optimizer_class=torch.optim.Adam)
    print("\n🚀 Running evaluation on the test set...")
    solver.evaluation(test_set)
    print("\n🎉 Evaluation complete.")


def parse_arguments():
    p = argparse.ArgumentParser(description="Baseline Model Evaluation Script")
    # --- Essential Arguments ---
    p.add_argument('--config', type=argparse.FileType(mode='r'), required=True, help='Path to the model config file.')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to the pre-trained model checkpoint file.')

    args, unknown = p.parse_known_args()

    # Load args from config file
    data = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in data.items():
        arg_dict[key] = value

    # --- Set Defaults for Missing Config Values ---
    defaults = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 123,
        'max_length': 6000,
        'key_format': 'fasta_descriptor',
        'embedding_mode': 'lm',
        'unknown_solubility': True,
        'batch_size': 128,
        'exp_name': 'baseline_eval',
        'optimizer_parameters': {'lr': 1e-4},
        'loss_function': 'LocCrossEntropy'
    }

    for key, value in defaults.items():
        if key not in arg_dict:
            arg_dict[key] = value

    return args

if __name__ == '__main__':
    parsed_args = parse_arguments()
    evaluate(parsed_args)
