import fire
import random as rnd
import sys
import os
import platform
import torch
from big_sleep import Imagine, version
from pathlib import Path
from .version import __version__;

def check_environment():
    """Check if the environment is properly set up for big-sleep"""
    print("╔══════════════════════════════════════════════════════╗")
    print("║                Big Sleep Environment                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    # Check if PyTorch is installed
    if not hasattr(torch, '__version__'):
        print("❌ ERROR: PyTorch is not properly installed.")
        print("   Please install PyTorch with: pip install torch torchvision")
        sys.exit(1)
    
    # Check for Apple Silicon specific requirements
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if is_apple_silicon:
        if not torch.backends.mps.is_available():
            print("⚠️  Running on Apple Silicon, but MPS is not available.")
            print("   Performance will be significantly slower on CPU.")
            print("   Make sure you have PyTorch 2.0+ installed: pip install 'torch>=2.0.0' 'torchvision>=0.15.0'")
        else:
            print(f"✅ Using Apple MPS (Metal Performance Shaders) for accelerated processing")
            print(f"   Hardware: {platform.processor()} - {platform.machine()}")
    elif torch.cuda.is_available():
        print(f"✅ Using CUDA with {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected. Running on CPU will be very slow.")
    
    print(f"• PyTorch version: {torch.__version__}")
    print(f"• Python version: {platform.python_version()}")
    print("────────────────────────────────────────────────────────")


def train(
    text=None,
    img=None,
    text_min="",
    lr = .07,
    image_size = 512,
    gradient_accumulate_every = 1,
    epochs = 20,
    iterations = 1050,
    save_every = 50,
    overwrite = False,
    save_progress = False,
    save_date_time = False,
    bilinear = False,
    open_folder = True,
    seed = 0,
    append_seed = False,
    random = False,
    torch_deterministic = False,
    max_classes = None,
    class_temperature = 2.,
    save_best = True,  # Changed to True to save best result by default
    experimental_resample = False,
    ema_decay = 0.5,
    num_cutouts = 96,  # Reduced from 128 for better performance
    center_bias = False,  # Matching original default
    larger_model = False,
    output_dir = None,
    fast = False,  # New parameter for quickly generating images
    debug = False   # New parameter to enable debug output
):
    print(f'Starting up... v{__version__}')

    if random:
        seed = rnd.randint(0, 1e6)
        
    # Handle output directory
    if output_dir:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Images will be saved to: {os.path.abspath(output_dir)}")

    # Apply fast mode settings if enabled
    if fast:
        epochs = 5
        iterations = 500
        num_cutouts = 64
        print("⚡ Fast mode enabled - using reduced settings for quicker generation")
        
    # Set the debug flag in the big_sleep module
    import big_sleep.big_sleep
    big_sleep.big_sleep.DEBUG = debug
    
    if debug:
        print("🔍 Debug mode enabled - verbose output will be shown")
        
    imagine = Imagine(
        text=text,
        img=img,
        text_min=text_min,
        lr = lr,
        image_size = image_size,
        gradient_accumulate_every = gradient_accumulate_every,
        epochs = epochs,
        iterations = iterations,
        save_every = save_every,
        save_progress = save_progress,
        bilinear = bilinear,
        seed = seed,
        append_seed = append_seed,
        torch_deterministic = torch_deterministic,
        open_folder = open_folder,
        max_classes = max_classes,
        class_temperature = class_temperature,
        save_date_time = save_date_time,
        save_best = save_best,
        experimental_resample = experimental_resample,
        ema_decay = ema_decay,
        num_cutouts = num_cutouts,
        center_bias = center_bias,
        larger_clip = larger_model,
        output_dir = output_dir
    )

    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            exit()

    imagine()
    
    # Print completion message
    print("\n╔════════════════════════════════════════════════════╗")
    print("║               Generation Complete!                  ║") 
    print("╚════════════════════════════════════════════════════╝")
    abs_path = os.path.abspath(str(imagine.filename))
    print(f"Image saved to: {abs_path}")
    print("────────────────────────────────────────────────────────")
    
    # Output directory is now handled by the Imagine class

def main():
    check_environment()
    fire.Fire(train)
