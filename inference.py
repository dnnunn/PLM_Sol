import copy

from models import *  # For loading classes specified in config
# from models.legacy import *  # For loading classes specified in config
from torch.optim import *  # For loading optimizer class that was used in the checkpoint
import os
import argparse
import yaml
import torch.nn as nn
from torchvision.transforms import transforms
from datasets.embeddings_dataset import Embeddings_predict_Dataset
from datasets.server_embeddings_dataset import ServerEmbeddings_predict_Dataset
from datasets.transforms import *
from solver import Solver
from models.biLSTM_TextCNN import biLSTM_TextCNN


def inference(args, server_embeddings=None):
    print(f"[DEBUG] Entered inference. Using server_embeddings: {server_embeddings is not None}")
    print(f"[DEBUG] Args: {args}")
    try:
        # Log config file if present
        if hasattr(args, 'config') and getattr(args, 'config', None):
            print(f"[DEBUG] Config file: {getattr(args, 'config', None)}")
        if hasattr(args, 'output_files_name'):
            print(f"[DEBUG] Output file will be: {args.output_files_name}")
    except Exception as e:
        print(f"[DEBUG] Exception printing args/config: {e}")
    """
    PLM_Sol inference with optional server embeddings support.
    
    Args:
        args: Configuration arguments
        server_embeddings: Optional list of pre-computed embeddings from server
                          If provided, skips embedding generation for 10-20x speedup
    """
    transform = transforms.Compose([Solubility_predict_ToInt(), predict_ToTensor()])

    # Use server embeddings if provided, otherwise load from file
    if server_embeddings is not None:
        # Use server-provided embeddings (fast path)
        data_set = ServerEmbeddings_predict_Dataset(
            server_embeddings=server_embeddings,
            remapped_sequences=args.remapping,
            key_format=args.key_format,
            embedding_mode=args.embedding_mode,
            transform=transform
        )
    else:
        # Traditional path: load embeddings from H5 file (slow)
        data_set = Embeddings_predict_Dataset(
            args.embeddings, args.remapping,
            key_format=args.key_format,
            embedding_mode=args.embedding_mode,
            transform=transform
        )
    
    model: nn.Module = globals()[args.model_type](embeddings_dim=data_set[0][0].shape[-1], **args.model_parameters)

    # Needs "from torch.optim import *" and "from models import *" to work
    solver = Solver(model, args, globals()[args.optimizer])
    
    # CRITICAL FIX: Save predictions to the expected output file
    # Use the exact output_files_name path provided (already includes .csv extension)
    output_filename = getattr(args, 'output_files_name', 'solubility_predictor_results.csv')
    print(f"[DEBUG] About to call solver.predict_evaluation with output_filename: {output_filename}")
    try:
        result = solver.predict_evaluation(data_set, filename=output_filename)
        print(f"[DEBUG] solver.predict_evaluation returned. Checking output file: {output_filename}")
        if os.path.exists(output_filename):
            print(f"[DEBUG] Output file {output_filename} exists. Size: {os.path.getsize(output_filename)} bytes")
        else:
            print(f"[DEBUG] Output file {output_filename} does NOT exist after prediction!")
        return result
    except Exception as e:
        print(f"[DEBUG] Exception in solver.predict_evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference.yaml')
    p.add_argument('--checkpoints_list', default=[],
                   help='if there are paths specified here, they all are evaluated')
    p.add_argument('--batch_size', type=int, default=16, help='samples that will be processed in parallel')
    p.add_argument('--log_iterations', type=int, default=100, help='log every log_iterations (-1 for no logging)')
    p.add_argument('--embeddings', type=str, default='data/embeddings/val_reduced.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--remapping', type=str, default='data/embeddings/val_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--distance_threshold', type=float, default=-1.0,
                   help='cutoff similarity for when to do lookup and when to use denovo predictions. If negative, denovo predictions will always be used.')
    p.add_argument('--key_format', type=str, default='hash',
                   help='the formatting of the keys in the h5 file [fasta_descriptor_old, fasta_descriptor, hash]')


    args = p.parse_args()
    arg_dict = args.__dict__
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


if __name__ == '__main__':
    original_args = copy.copy(parse_arguments())
    
    # CRITICAL FIX: If no checkpoints_list provided, run inference directly with config args
    if not original_args.checkpoints_list:
        print(f"[DEBUG] No checkpoints_list provided, running inference with config args directly")
        inference(original_args)
    else:
        # Original logic for multiple checkpoints
        for checkpoint in original_args.checkpoints_list:
            args = copy.copy(original_args)
            arg_dict = args.__dict__
            arg_dict['checkpoint'] = checkpoint
            # get the arguments from the yaml config file that is saved in the runs checkpoint
            data = yaml.load(open(os.path.join('./model_param/train_arguments.yml'), 'r'), Loader=yaml.FullLoader)
            for key, value in data.items():
                if key not in args.__dict__.keys():
                    if isinstance(value, list):
                        for v in value:
                            arg_dict[key].append(v)
                    else:
                        arg_dict[key] = value
            # call teh actual inference
            inference(args)
   
