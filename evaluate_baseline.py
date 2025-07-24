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

    # 1. Load the test dataset
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
        checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
        # The checkpoint might be a dictionary, check for 'model_state_dict' or similar keys
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"💾 Successfully loaded baseline model weights from: {args.checkpoint}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # 4. Set up the solver and run evaluation
    # The optimizer is not needed for evaluation, so we can pass a placeholder.
    solver = Solver(model, args, optimizer_class=torch.optim.Adam)
    print("\n🚀 Running evaluation on the test set...")
    solver.evaluation(test_set)
    print("\n🎉 Evaluation complete.")


def parse_arguments():
    p = argparse.ArgumentParser(description="Baseline Model Evaluation Script")
    # --- Essential Arguments ---
    p.add_argument('--config', type=argparse.FileType(mode='r'), required=True, help='Path to the model config file (e.g., the one used for fine-tuning).')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to the pre-trained model checkpoint file.')
    p.add_argument('--test_embeddings', type=str, required=True, help='.h5 file for the test set.')
    p.add_argument('--test_remapping', type=str, required=True, help='Fasta remapping file for the test set.')
    
    # --- Defaultable Arguments (from train.py) ---
    p.add_argument('--seed', type=int, default=123, help='Seed for reproducibility.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on.')
    p.add_argument('--model_type', type=str, default='FFN', help='Classname of the model.')
    p.add_argument('--model_parameters', type=dict, help='Dictionary of model parameters.')
    p.add_argument('--loss_function', type=str, default='LocCrossEntropy', help='Loss function class name.')
    p.add_argument('--unknown_solubility', type=bool, default=True, help='Include sequences with unknown solubility.')
    p.add_argument('--max_length', type=int, default=6000, help='Maximum sequence length.')
    p.add_argument('--embedding_mode', type=str, default='lm', help='Type of embedding to use.')
    p.add_argument('--key_format', type=str, default='fasta_descriptor', help='Key format in the h5 file.')
    p.add_argument('--exp_name', type=str, default='baseline_eval', help='Experiment name for solver.')
    p.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    p.add_argument('--optimizer_parameters', type=dict, default={'lr': 1e-4}, help='Dummy optimizer params.')

    args = p.parse_args()
    
    # Load args from config file, overriding defaults
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if key != 'checkpoint' and key != 'test_embeddings' and key != 'test_remapping': # Don't override essential args
                arg_dict[key] = value
    return args

if __name__ == '__main__':
    parsed_args = parse_arguments()
    evaluate(parsed_args)
