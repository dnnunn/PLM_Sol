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

    # 1. Load the test dataset, allowing CLI override for validation or other sets
    eval_embeddings = getattr(args, 'eval_embeddings', None)
    eval_remapping = getattr(args, 'eval_remapping', None)
    if eval_embeddings and eval_remapping:
        print(f"\n📖 Loading evaluation set from CLI override:")
        print(f"   - Embeddings: {eval_embeddings}")
        print(f"   - Remapping: {eval_remapping}")
        embeddings_path = eval_embeddings
        remapping_path = eval_remapping
    else:
        print(f"\n📖 Loading test set from config paths:")
        print(f"   - Embeddings: {args.test_embeddings}")
        print(f"   - Remapping: {args.test_remapping}")
        embeddings_path = args.test_embeddings
        remapping_path = args.test_remapping
    transform = transforms.Compose([SolubilityToInt(), ToTensor()])
    test_set = EmbeddingsDataset(embeddings_path, remapping_path, args.unknown_solubility,
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
            result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Model loading result (model_state_dict):", result)
        else:
            result = model.load_state_dict(checkpoint, strict=False)
            print("Model loading result (raw state_dict):", result)
        print(f"💾 Successfully loaded baseline model weights from: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # 4. Set up the solver and run evaluation
    solver = Solver(model, args)
    print("\n🚀 Running evaluation on the test set...")
    solver.evaluation(test_set)
    print("\n🎉 Evaluation complete.")


def parse_arguments():
    p = argparse.ArgumentParser(description="Baseline Model Evaluation Script")
    # --- Essential Arguments ---
    p.add_argument('--config', type=argparse.FileType(mode='r'), required=True, help='Path to the fine-tuning config file (for data paths).')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to the pre-trained model checkpoint file.')
    # --- Optional Evaluation Overrides ---
    p.add_argument('--eval_embeddings', type=str, default=None, help='Override embeddings file for evaluation (validation or other set)')
    p.add_argument('--eval_remapping', type=str, default=None, help='Override remapping file for evaluation (validation or other set)')

    args, unknown = p.parse_known_args()

    # Save the command-line checkpoint before it gets overwritten by the config
    cmd_checkpoint = args.checkpoint

    # Load data paths from the fine-tuning config
    data = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in data.items():
        arg_dict[key] = value

    # Restore the command-line checkpoint, which was overwritten by the config
    args.checkpoint = cmd_checkpoint

    # --- Override with correct baseline model architecture and parameters ---
    print("\n🔧 Overriding model architecture to match baseline checkpoint (biLSTM_TextCNN).")
    args.model_type = 'biLSTM_TextCNN'
    args.model_parameters = {
        'output_dim': 1,
        'dropout': 0.25,
        'kernel_size': 9,
        'conv_dropout': 0.25
    }

    # --- Set Defaults for any other missing values ---
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
