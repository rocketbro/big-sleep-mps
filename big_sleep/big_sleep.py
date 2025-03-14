import os
import os.path
import sys
import subprocess
import signal
import string
import re

from datetime import datetime
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image as _save_image

# Global debug flag (set via CLI)
DEBUG = False

# Custom save image function with optional debugging
def save_image(img, fp):
    try:
        if DEBUG:
            print(f"DEBUG SAVE: Saving image to {fp}, shape={img.shape}, dtype={img.dtype}")
        _save_image(img, fp)
        if DEBUG:
            if os.path.exists(fp):
                print(f"DEBUG SAVE: File exists after save: {fp}, size={os.path.getsize(fp)}")
            else:
                print(f"DEBUG SAVE: File does not exist after save: {fp}")
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        if DEBUG:
            import traceback
            traceback.print_exc()
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm, trange

from big_sleep.ema import EMA
from big_sleep.resample import resample
from big_sleep.biggan import BigGAN
from big_sleep.clip import load, tokenize

# Check if we're running on Apple Silicon
import platform
is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'

# Use MPS (Metal Performance Shaders) if on Apple Silicon, otherwise require CUDA
if is_apple_silicon:
    if torch.backends.mps.is_available():
        try:
            # Create a small tensor to test if MPS is working properly
            test_tensor = torch.zeros(1, device='mps')
            # If we got here, MPS seems to be working
            DEVICE = torch.device('mps')
            print('✅ MPS is functioning properly')
        except Exception as e:
            print(f'⚠️ MPS is available but encountered an error: {str(e)}')
            print('⚠️ Falling back to CPU')
            DEVICE = torch.device('cpu')
    else:
        print('⚠️ MPS is not available on this Apple Silicon device')
        DEVICE = torch.device('cpu')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    print('WARNING: Running on CPU, which will be very slow. Consider using a GPU.')

# graceful keyboard interrupt

terminate = False

def signal_handling(signum,frame):
    print('detecting keyboard interrupt, gracefully exiting')
    global terminate
    terminate = True

signal.signal(signal.SIGINT,signal_handling)

# helpers

def exists(val):
    return val is not None

def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/','\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def create_text_path(text=None, img=None, encoding=None):
    input_name = ""
    if text is not None:
        input_name += text
    if img is not None:
        if isinstance(img, str):
            img_name = "".join(img.split(".")[:-1]) # replace spaces by underscores, remove img extension
            img_name = img_name.split("/")[-1]  # only take img name, not path
        else:
            img_name = "PIL_img"
        input_name += "_" + img_name
    if encoding is not None:
        input_name = "your_encoding"
    return input_name.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:255]

# tensor helpers

def differentiable_topk(x, k, temperature=1.):
    n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x = x.scatter(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(n, k, dim).sum(dim = 1)


def create_clip_img_transform(image_width):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
                    #T.ToPILImage(),
                    T.Resize(image_width),
                    T.CenterCrop((image_width, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=clip_mean, std=clip_std)
            ])
    return transform


def rand_cutout(image, size, center_bias=False, center_focus=2):
    width = image.shape[-1]
    min_offset = 0
    max_offset = width - size
    if center_bias:
        # sample around image center
        center = max_offset / 2
        std = center / center_focus
        offset_x = int(random.gauss(mu=center, sigma=std))
        offset_y = int(random.gauss(mu=center, sigma=std))
        # resample uniformly if over boundaries
        offset_x = random.randint(min_offset, max_offset) if (offset_x > max_offset or offset_x < min_offset) else offset_x
        offset_y = random.randint(min_offset, max_offset) if (offset_y > max_offset or offset_y < min_offset) else offset_y
    else:
        offset_x = random.randint(min_offset, max_offset)
        offset_y = random.randint(min_offset, max_offset)
    cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]
    return cutout

# load biggan

class Latents(torch.nn.Module):
    def __init__(
        self,
        num_latents = 15,
        num_classes = 1000,
        z_dim = 128,
        max_classes = None,
        class_temperature = 2.
    ):
        super().__init__()
        self.normu = torch.nn.Parameter(torch.zeros(num_latents, z_dim).normal_(std = 1))
        self.cls = torch.nn.Parameter(torch.zeros(num_latents, num_classes).normal_(mean = -3.9, std = .3))
        self.register_buffer('thresh_lat', torch.tensor(1))

        assert not exists(max_classes) or max_classes > 0 and max_classes <= num_classes, f'max_classes must be between 0 and {num_classes}'
        self.max_classes = max_classes
        self.class_temperature = class_temperature

    def forward(self):
        if exists(self.max_classes):
            classes = differentiable_topk(self.cls, self.max_classes, temperature = self.class_temperature)
        else:
            classes = torch.sigmoid(self.cls)

        return self.normu, classes

class Model(nn.Module):
    def __init__(
        self,
        image_size,
        max_classes = None,
        class_temperature = 2.,
        ema_decay = 0.99
    ):
        super().__init__()
        assert image_size in (128, 256, 512), 'image size must be one of 128, 256, or 512'
        self.biggan = BigGAN.from_pretrained(f'biggan-deep-{image_size}')
        self.max_classes = max_classes
        self.class_temperature = class_temperature
        self.ema_decay\
            = ema_decay

        self.init_latents()

    def init_latents(self):
        latents = Latents(
            num_latents = len(self.biggan.config.layers) + 1,
            num_classes = self.biggan.config.num_classes,
            z_dim = self.biggan.config.z_dim,
            max_classes = self.max_classes,
            class_temperature = self.class_temperature
        )
        self.latents = EMA(latents, self.ema_decay)

    def forward(self):
        self.biggan.eval()
        out = self.biggan(*self.latents(), 1)
        return (out + 1) / 2


class BigSleep(nn.Module):
    def __init__(
        self,
        num_cutouts = 128,
        loss_coef = 100,
        image_size = 512,
        bilinear = False,
        max_classes = None,
        class_temperature = 2.,
        experimental_resample = False,
        ema_decay = 0.99,
        center_bias = False,
        larger_clip = False
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_size = image_size
        self.num_cutouts = num_cutouts
        self.experimental_resample = experimental_resample
        self.center_bias = center_bias

        self.interpolation_settings = {'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

        model_name = 'ViT-B/32' if not larger_clip else 'ViT-L/14'
        self.perceptor, self.normalize_image = load(model_name, device=DEVICE, jit=False)

        self.model = Model(
            image_size = image_size,
            max_classes = max_classes,
            class_temperature = class_temperature,
            ema_decay = ema_decay
        )

    def reset(self):
        self.model.init_latents()

    def sim_txt_to_img(self, text_embed, img_embed, text_type="max"):
        sign = -1
        if text_type == "min":
            sign = 1
        return sign * self.loss_coef * torch.cosine_similarity(text_embed, img_embed, dim = -1).mean()

    def forward(self, text_embeds, text_min_embeds=[], return_loss = True):
        width, num_cutouts = self.image_size, self.num_cutouts

        out = self.model()

        if not return_loss:
            return out

        pieces = []
        for ch in range(num_cutouts):
            # sample cutout size
            size = int(width * torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
            # get cutout
            apper = rand_cutout(out, size, center_bias=self.center_bias)
            if (self.experimental_resample):
                apper = resample(apper, (224, 224))
            else:
                apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
            pieces.append(apper)

        into = torch.cat(pieces)
        into = self.normalize_image(into)

        image_embed = self.perceptor.encode_image(into)

        latents, soft_one_hot_classes = self.model.latents()
        num_latents = latents.shape[0]
        latent_thres = self.model.latents.model.thresh_lat

        lat_loss =  torch.abs(1 - torch.std(latents, dim=1)).mean() + \
                    torch.abs(torch.mean(latents, dim = 1)).mean() + \
                    4 * torch.max(torch.square(latents).mean(), latent_thres)


        for array in latents:
            mean = torch.mean(array)
            diffs = array - mean
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            skews = torch.mean(torch.pow(zscores, 3.0))
            kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

            lat_loss = lat_loss + torch.abs(kurtoses) / num_latents + torch.abs(skews) / num_latents

        cls_loss = ((50 * torch.topk(soft_one_hot_classes, largest = False, dim = 1, k = 999)[0]) ** 2).mean()

        results = []
        for txt_embed in text_embeds:
            results.append(self.sim_txt_to_img(txt_embed, image_embed))
        for txt_min_embed in text_min_embeds:
            results.append(self.sim_txt_to_img(txt_min_embed, image_embed, "min"))
        sim_loss = sum(results).mean()
        return out, (lat_loss, cls_loss, sim_loss)


class Imagine(nn.Module):
    def __init__(
        self,
        *,
        text=None,
        img=None,
        encoding=None,
        text_min = "",
        lr = .07,
        image_size = 512,
        gradient_accumulate_every = 1,
        save_every = 50,
        epochs = 20,
        iterations = 1050,
        save_progress = False,
        bilinear = False,
        open_folder = True,
        seed = None,
        append_seed = False,
        torch_deterministic = False,
        max_classes = None,
        class_temperature = 2.,
        save_date_time = False,
        save_best = False,
        experimental_resample = False,
        ema_decay = 0.99,
        num_cutouts = 128,
        center_bias = False,
        larger_clip = False,
        output_dir = None
    ):
        super().__init__()

        if torch_deterministic:
            assert not bilinear, 'the deterministic (seeded) operation does not work with interpolation (PyTorch 1.7.1)'
            torch.set_deterministic(True)

        self.seed = seed
        self.append_seed = append_seed

        if exists(seed):
            print(f'setting seed of {seed}')
            if seed == 0:
                print('you can override this with --seed argument in the command line, or --random for a randomly chosen one')
            torch.manual_seed(seed)

        self.epochs = epochs
        self.iterations = iterations

        model = BigSleep(
            image_size = image_size,
            bilinear = bilinear,
            max_classes = max_classes,
            class_temperature = class_temperature,
            experimental_resample = experimental_resample,
            ema_decay = ema_decay,
            num_cutouts = num_cutouts,
            center_bias = center_bias,
            larger_clip = larger_clip
        ).to(DEVICE)

        self.model = model

        self.lr = lr
        self.optimizer = Adam(model.model.latents.model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.save_date_time = save_date_time

        self.save_best = save_best
        self.current_best_score = 0

        self.open_folder = open_folder
        self.total_image_updates = (self.epochs * self.iterations) / self.save_every
        self.encoded_texts = {
            "max": [],
            "min": []
        }
        # create img transform
        self.clip_transform = create_clip_img_transform(224)
        # create starting encoding
        self.output_dir = output_dir  # Store output_dir as a class property
        self.set_clip_encoding(text=text, img=img, encoding=encoding, text_min=text_min)
    
    @property
    def seed_suffix(self):
        return f'.{self.seed}' if self.append_seed and exists(self.seed) else ''

    def set_text(self, text):
        self.set_clip_encoding(text = text)

    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if encoding is not None:
            encoding = encoding.to(DEVICE)
        #elif self.create_story:
        #    encoding = self.update_story_encoding(epoch=0, iteration=1)
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).to(DEVICE)
        with torch.no_grad():
            text_encoding = self.model.perceptor.encode_text(tokenized_text).detach()
        return text_encoding
    
    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_encoding = self.model.perceptor.encode_image(normed_img).detach()
        return img_encoding
    
    
    def encode_multiple_phrases(self, text, img=None, encoding=None, text_type="max"):
        if text is not None and "|" in text:
            self.encoded_texts[text_type] = [self.create_clip_encoding(text=prompt_min, img=img, encoding=encoding) for prompt_min in text.split("|")]
        else:
            self.encoded_texts[text_type] = [self.create_clip_encoding(text=text, img=img, encoding=encoding)]

    def encode_max_and_min(self, text, img=None, encoding=None, text_min=""):
        self.encode_multiple_phrases(text, img=img, encoding=encoding)
        if text_min is not None and text_min != "":
            self.encode_multiple_phrases(text_min, img=img, encoding=encoding, text_type="min")

    def set_clip_encoding(self, text=None, img=None, encoding=None, text_min=""):
        self.current_best_score = 0
        self.text = text
        self.text_min = text_min
        
        if len(text_min) > 0:
            text = text + "_wout_" + text_min[:255] if text is not None else "wout_" + text_min[:255]
        text_path = create_text_path(text=text, img=img, encoding=encoding)
        if self.save_date_time:
            text_path = datetime.now().strftime("%y%m%d-%H%M%S-") + text_path

        self.text_path = text_path
        
        # Handle output directory if specified
        if self.output_dir:
            output_path = Path(self.output_dir)
            os.makedirs(output_path, exist_ok=True)
            self.filename = output_path / f'{text_path}{self.seed_suffix}.png'
        else:
            self.filename = Path(f'./{text_path}{self.seed_suffix}.png')
            
        self.encode_max_and_min(text, img=img, encoding=encoding, text_min=text_min) # Tokenize and encode each prompt

    def reset(self):
        self.model.reset()
        self.model = self.model.to(DEVICE)
        self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    def train_step(self, epoch, i, pbar=None):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            out, losses = self.model(self.encoded_texts["max"], self.encoded_texts["min"])
            loss = sum(losses) / self.gradient_accumulate_every
            total_loss += loss
            loss.backward()

        self.optimizer.step()
        self.model.model.latents.update()
        self.optimizer.zero_grad()

        if (i + 1) % self.save_every == 0:
            with torch.no_grad():
                self.model.model.latents.eval()
                out, losses = self.model(self.encoded_texts["max"], self.encoded_texts["min"])
                top_score, best = torch.topk(losses[2], k=1, largest=False)
                image = self.model.model()[best].cpu()
                self.model.model.latents.train()

                # Get absolute path for clarity
                abs_path = os.path.abspath(str(self.filename))
                if DEBUG:
                    print(f"DEBUG: Saving image to path: {abs_path}")
                
                # Make sure the directory exists
                out_dir = os.path.dirname(abs_path)
                if DEBUG:
                    print(f"DEBUG: Creating directory: {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                
                # Debug image info
                if DEBUG:
                    print(f"DEBUG: Image shape: {image.shape}, type: {image.dtype}")
                
                # Save the image
                try:
                    save_image(image, str(self.filename))
                    if DEBUG:
                        print(f"DEBUG: Successfully saved image")
                except Exception as e:
                    print(f"ERROR: Failed to save image: {str(e)}")
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
                    
                if pbar is not None:
                    pbar.update(1)
                else:
                    print(f'Image saved to: {abs_path}')

                if self.save_progress:
                    total_iterations = epoch * self.iterations + i
                    num = total_iterations // self.save_every
                    if self.output_dir:
                        progress_path = Path(self.output_dir) / f'{self.text_path}.{num}{self.seed_suffix}.png'
                    else:
                        progress_path = Path(f'./{self.text_path}.{num}{self.seed_suffix}.png')
                    save_image(image, progress_path)

                if self.save_best and top_score.item() < self.current_best_score:
                    self.current_best_score = top_score.item()
                    if self.output_dir:
                        best_path = Path(self.output_dir) / f'{self.text_path}{self.seed_suffix}.best.png'
                    else:
                        best_path = Path(f'./{self.text_path}{self.seed_suffix}.best.png')
                    save_image(image, best_path)

        return out, total_loss

    def forward(self):
        penalizing = ""
        if len(self.text_min) > 0:
            penalizing = f'penalizing "{self.text_min}"'
        
        print("\n╔════════════════════════════════════════════════════╗")
        print(f"║ Generating: \"{self.text_path}\"")
        if penalizing:
            print(f"║ {penalizing}")
        print("╚════════════════════════════════════════════════════╝")
        
        # Display configuration
        print(f"• Image size: {self.model.image_size}x{self.model.image_size}")
        print(f"• Iterations per epoch: {self.iterations}")
        print(f"• Total epochs: {self.epochs}")
        print(f"• Output will be saved to: {self.filename}")
        print(f"• Device: {DEVICE}")
        print("────────────────────────────────────────────────────────")
        
        # Ensure output directory exists
        if self.output_dir:
            os.makedirs(os.path.dirname(str(self.filename)), exist_ok=True)
        
        with torch.no_grad():
            print("Warming up model...")
            self.model(self.encoded_texts["max"][0]) # one warmup step due to issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        image_pbar = tqdm(total=self.total_image_updates, desc='image update', position=2, leave=True)
        epoch_pbar = trange(self.epochs, desc = '      epochs', position=0, leave=True)
        for epoch in (ep for ep in epoch_pbar if not terminate):
            pbar = trange(self.iterations, desc='   iteration', position=1, leave=True)
            image_pbar.update(0)
            for i in (it for it in pbar if not terminate):
                out, loss = self.train_step(epoch, i, image_pbar)
                pbar.set_description(f'loss: {loss.item():04.2f}')

