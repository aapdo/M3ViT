"""
Utilities for model tracing, debugging, and forward hooks.
"""
import logging
import threading
from pathlib import Path
import functools
import inspect

import torch
import torch.utils.data as data

class DatasetWithIndex(data.Dataset):
    """Wrapper to return (index, data) instead of just data.
    Used only when forward_hook is enabled for debugging."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        targets = sample.copy()
        image = {'image': targets.pop('image')}
        # Keep 'meta' as it's needed for evaluation
        meta = {}
        if 'meta' in targets:
            meta = {'meta': targets.pop('meta')}
        return (idx, {**image, **meta}, targets)

    def __len__(self):
        return len(self.dataset)


def handle_forward_hook_data(args, data):
    """
    Unpacks data for forward_hook, updates batch_counter, and returns a unified data dictionary.

    Args:
        args: Argparse namespace. Must have 'forward_hook' and 'batch_counter' attributes.
        data: Data from the DataLoader. Can be a dictionary or a tuple of
              (indices, samples_dict, targets_dict) if forward_hook is enabled.

    Returns:
        dict: A unified data dictionary containing samples and targets.
    """
    if hasattr(args, 'forward_hook') and args.forward_hook:
        # data is [indices, samples_dict, targets_dict]
        indices, samples_dict, targets_dict = data
        # Update batch_counter with dataset indices
        if hasattr(args, 'batch_counter'):
            args.batch_counter['current_indices'] = indices.tolist()
        # Reconstruct data as a single dictionary
        return {**samples_dict, **targets_dict}
    return data


def setup_forward_hooks(model, args, log_directory, criterion=None):
    """
    Sets up forward hooks on model layers and loss criterion to log activations for debugging.

    Args:
        model: The model to attach hooks to.
        args: Argparse namespace with 'rank'.
        log_directory (str): The directory to save the hook log file.
        criterion: Optional loss criterion module to attach hooks to.

    Returns:
        tuple: A tuple containing:
            - list: A list of hook handles.
            - dict: The batch_counter dictionary for tracking indices.
    """
    hook_handles = []
    # Setup separate logger for forward hooks (all ranks write to same file)
    hook_logger = logging.getLogger('forward_hook')
    hook_logger.setLevel(logging.INFO)
    hook_logger.handlers = []  # Clear any existing handlers

    # Create forward_hook.log file
    hook_log_dir = Path(log_directory)
    hook_log_dir.mkdir(parents=True, exist_ok=True)
    hook_log_file = hook_log_dir / 'forward_hook.log'

    # Clear the file at the start (rank 0 only)
    rank = getattr(args, 'rank', 0)
    if rank == 0:
        with open(hook_log_file, 'w', encoding='utf-8') as f:
            pass  # Just clear the file
        print(f"Cleared forward_hook.log")

    # All ranks write to the same file (append mode to avoid overwriting)
    hook_file_handler = logging.FileHandler(hook_log_file, mode='a', encoding='utf-8')
    hook_file_handler.setLevel(logging.INFO)
    hook_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    hook_file_handler.setFormatter(hook_formatter)
    hook_logger.addHandler(hook_file_handler)

    # Don't add console handler to avoid output to log_file.txt
    # (since sys.stdout and sys.stderr are redirected to Logger)

    if rank == 0:
        print(f"Forward hook logging to: {hook_log_file}")

    # Track dataset indices globally (shared across all layers)
    # Engine_mt.py will update current_indices before each forward pass
    batch_counter = {
        'current_indices': [],  # Store dataset indices for current batch
        'enabled': True         # Control whether hooks are active
    }

    # Hook lock to prevent log interleaving
    hook_lock = threading.Lock()

    def make_forward_hook(layer_name, log_input=True, log_output=True):
        def hook(module, input, output):
            # Skip if hooks are disabled
            if not batch_counter.get('enabled', True):
                return

            # Use global rank for consistency
            gpu_id = getattr(args, 'rank', 0)

            with hook_lock:
                # Get dataset indices if available
                dataset_indices_list = batch_counter['current_indices']
                indices_log_part = ""
                if dataset_indices_list:
                    indices_log_part = f" dataset_idx: {dataset_indices_list}"

                # Print input (first 100 values only)
                if log_input:
                    if isinstance(input, tuple) and len(input) > 0:
                        inp = input[0]
                        if isinstance(inp, torch.Tensor):
                            inp_values = inp.flatten().tolist()[:100]
                            hook_logger.info(f"[GPU{gpu_id}]{indices_log_part} layer_name: {layer_name}, INPUT_shape: {tuple(inp.shape)}, INPUT_values: {inp_values}")

                # hook_logger.info output (first 100 values only)
                if log_output:
                    if isinstance(output, torch.Tensor):
                        out_values = output.flatten().tolist()[:100]
                        hook_logger.info(f"[GPU{gpu_id}]{indices_log_part} layer_name: {layer_name}, OUTPUT_shape: {tuple(output.shape)}, OUTPUT_values: {out_values}")
                    elif isinstance(output, (tuple, list)):
                        for idx, out in enumerate(output):
                            if isinstance(out, torch.Tensor):
                                out_values = out.flatten().tolist()[:100]
                                hook_logger.info(f"[GPU{gpu_id}]{indices_log_part} layer_name: {layer_name}[{idx}], OUTPUT_shape: {tuple(out.shape)}, OUTPUT_values: {out_values}")
        return hook

    # Register hooks only for specific layers
    hook_messages = []
    for name, module in model.named_modules():
        # Determine log settings based on layer type
        log_input = False
        log_output = False
        should_hook = False

        # Handle DDP wrapper: strip 'module.' prefix for checking
        name_to_check = name.replace('module.', '')

        # patch_embed: log both input and output
        if 'patch_embed' in name_to_check and name_to_check.endswith('patch_embed'):
            should_hook = True
            log_input = True
            log_output = True

        # block 0 norm1: log input only (with values)
        elif name_to_check == 'blocks.0.norm1' or name_to_check == 'backbone.blocks.0.norm1':
            should_hook = True
            log_input = True
            log_output = False

        # blocks 0, 1: attn.proj_drop and mlp - output only
        elif (name_to_check.startswith('blocks.0.') or name_to_check.startswith('blocks.1.') or
                name_to_check.startswith('backbone.blocks.0.') or name_to_check.startswith('backbone.blocks.1.')):
            if name_to_check.endswith('.attn.proj_drop') or name_to_check.endswith('.mlp'):
                should_hook = True
                log_input = False
                log_output = True

        # head conv_4: output only
        elif name_to_check.startswith('heads.') and name_to_check.endswith('.conv_4'):
            should_hook = True
            log_input = False
            log_output = True

        if should_hook:
            # Remove 'module.' prefix for cleaner logging
            clean_name = name.replace('module.', '')
            handle = module.register_forward_hook(make_forward_hook(clean_name, log_input, log_output))
            hook_handles.append(handle)
            hook_messages.append(f"  {clean_name} (input={log_input}, output={log_output})")

    # Register hooks for criterion (loss modules)
    if criterion is not None:
        for name, module in criterion.named_modules():
            module_class_name = module.__class__.__name__
            if 'Loss' in module_class_name:
                # Remove 'criterion.' prefix for cleaner logging
                clean_name = name if name else module_class_name
                handle = module.register_forward_hook(make_forward_hook(clean_name, log_input=True, log_output=True))
                hook_handles.append(handle)
                hook_messages.append(f"  {clean_name} (input=True, output=True)")

    # Output all hook registration messages at once (rank 0 only)
    if rank == 0:
        if hook_messages:
            print(f"[GPU{rank}] Registered {len(hook_handles)} hooks:\n" + "\n".join(hook_messages))
        else:
            print(f"[GPU{rank}] No hooks registered")

    return hook_handles, batch_counter


def wrap_datasets_for_forward_hook(dataset_train, dataset_val):
    """
    Wraps training and validation datasets with DatasetWithIndex.

    Args:
        dataset_train: The training dataset.
        dataset_val: The validation dataset.

    Returns:
        tuple: A tuple containing the wrapped training and validation datasets.
    """
    print("[Forward Hook] Wrapping datasets with index tracking.")
    return DatasetWithIndex(dataset_train), DatasetWithIndex(dataset_val)


def patch_and_log_initializations(modules_to_patch, args):
    """
    Monkey-patches __init__ methods of classes in given modules to log their creations.

    Args:
        modules_to_patch (list): A list of modules to patch.
        args: Argparse namespace with 'rank'.

    Returns:
        dict: A dictionary storing the original __init__ methods.
    """
    original_inits = {}
    log_lock = threading.Lock()

    def wrap_init(original_init, class_name, module_name):
        @functools.wraps(original_init)
        def wrapper(self, *init_args, **init_kwargs):
            rank = getattr(args, 'rank', 0)
            if rank == 0:
                with log_lock:
                    log_lines = ["="*80, f"Creating: {module_name}.{class_name}"]
                    try:
                        sig = inspect.signature(original_init)
                        param_names = list(sig.parameters.keys())[1:]
                    except:
                        param_names = []
                    
                    all_args = {}
                    for idx, arg in enumerate(init_args):
                        if idx < len(param_names):
                            all_args[param_names[idx]] = arg
                        else:
                            all_args[f"arg_{idx}"] = arg
                    all_args.update(init_kwargs)

                    if all_args:
                        log_lines.append("  Arguments:")
                        for key, value in all_args.items():
                            if isinstance(value, (list, tuple)) and len(str(value)) > 100:
                                log_lines.append(f"    {key}: <{type(value).__name__} with {len(value)} items>")
                            elif isinstance(value, torch.Tensor):
                                log_lines.append(f"    {key}: Tensor{tuple(value.shape)}")
                            elif isinstance(value, torch.nn.Module):
                                log_lines.append(f"    {key}: {type(value).__name__} module")
                            else:
                                log_lines.append(f"    {key}: {value}")
                    else:
                        log_lines.append("  Arguments: (none)")
                    log_lines.append("="*80)
                    print("\n".join(log_lines))
            return original_init(self, *init_args, **init_kwargs)
        return wrapper

    exclude_classes = {
        'Module', 'Sequential', 'ModuleList', 'ModuleDict', 'ParameterList',
        'ParameterDict', 'Container', 'Identity', 'Linear', 'Conv2d', 'BatchNorm2d',
        'LayerNorm', 'Dropout', 'ReLU', 'GELU', 'Softmax', 'LogSoftmax',
        'CrossEntropyLoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BCEWithLogitsLoss',
        'NLLLoss', 'KLDivLoss', 'SmoothL1Loss', 'CosineSimilarity', 'PairwiseDistance'
    }

    for module in modules_to_patch:
        for name in dir(module):
            if name in exclude_classes:
                continue
            obj = getattr(module, name)
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                    original_inits[f"{module.__name__}.{name}"] = obj.__init__
                    obj.__init__ = wrap_init(obj.__init__, name, module.__name__)
    
    return original_inits

def restore_original_initializations(modules_to_patch, original_inits):
    """
    Restores the original __init__ methods after monkey-patching.

    Args:
        modules_to_patch (list): A list of modules that were patched.
        original_inits (dict): A dictionary storing the original __init__ methods.
    """
    for module in modules_to_patch:
        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                key = f"{module.__name__}.{name}"
                if key in original_inits:
                    obj.__init__ = original_inits[key]
