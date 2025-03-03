<img src="./samples/artificial_intelligence.png" width="250px"></img>

*artificial intelligence*

<img src="./samples/cosmic_love_and_attention.png" width="250px"></img>

*cosmic love and attention*

<img src="./samples/fire_in_the_sky.png" width="250px"></img>

*fire in the sky*

<img src="./samples/a_pyramid_made_of_ice.png" width="250px"></img>

*a pyramid made of ice*

<img src="./samples/a_lonely_house_in_the_woods.png" width="250px"></img>

*a lonely house in the woods*

<img src="./samples/marriage_in_the_mountains.png" width="250px"></img>

*marriage in the mountains*

<img src="./samples/a_lantern_dangling_from_the_tree_in_a_foggy_graveyard.png" width="250px"></img>

*lantern dangling from a tree in a foggy graveyard*

<img src="./samples/a_vivid_dream.png" width="250px"></img>

*a vivid dream*

<img src="./samples/balloons_over_the_ruins_of_a_city.png" width="250px"></img>

*balloons over the ruins of a city*

<img src="./samples/the_death_of_the_lonesome_astronomer.png" width="250px"></img>

*the death of the lonesome astronomer* - by <a href="https://github.com/moirage">moirage</a>

<img src="./samples/the_tragic_intimacy_of_the_eternal_conversation_with_oneself.png" width="250px"></img>

*the tragic intimacy of the eternal conversation with oneself* - by <a href="https://github.com/moirage">moirage</a>

<img src="./samples/demon_fire.png" width="250px"></img>

*demon fire* - by <a href="https://github.com/WiseNat">WiseNat</a>

## Big Sleep

<a href="https://twitter.com/advadnoun">Ryan Murdock</a> has done it again, combining OpenAI's <a href="https://github.com/openai/CLIP">CLIP</a> and the generator from a <a href="https://arxiv.org/abs/1809.11096">BigGAN</a>! This repository wraps up his work so it is easily accessible to anyone who owns a GPU.

You will be able to have the GAN dream up images using natural language with a one-line command in the terminal.

Original notebook [![Open In Colab][colab-badge]][colab-notebook]

Simplified notebook [![Open In Colab][colab-badge]][colab-notebook-2]

User-made notebook with bugfixes and added features, like google drive integration [![Open In Colab][colab-badge]][user-made-colab-notebook]

[user-made-colab-notebook]: <https://colab.research.google.com/drive/1zVHK4t3nXQTsu5AskOOOf3Mc9TnhltUO?usp=sharing>
[colab-notebook]: <https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing>
[colab-notebook-2]: <https://colab.research.google.com/drive/1MEWKbm-driRNF8PrU7ogS5o3se-ePyPb?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

## Install

```bash
$ pip install big-sleep
```

## Usage

```bash
$ dream "a pyramid made of ice"
```

Images will be saved to the current directory. You can specify a different output directory:

```bash
$ dream "a pyramid made of ice" --output-dir ~/Documents/my_images
```

## Apple Silicon Support

This version of Big Sleep has been updated to work natively on Apple Silicon (M1/M2/M3) Macs using the Metal Performance Shaders (MPS) backend in PyTorch. No external GPU is required to run on Apple Silicon devices.

When you install big-sleep on an Apple Silicon Mac, the appropriate PyTorch version (2.0+) with MPS support will be automatically installed for you.

Simply run:
```bash
$ pip install big-sleep
$ dream "your prompt"
```

The code will automatically detect Apple Silicon and use the appropriate device. No additional configuration is needed.

### Fast Mode for Quick Generation

To generate images more quickly (at the cost of some quality), use the `--fast` flag:

```bash
$ dream "cosmic explosion" --fast
```

This mode uses fewer iterations and epochs to produce results faster, which is especially helpful for testing prompts or on slower hardware.

For better composition with a central focus point, you can also try the `--center-bias` flag:

```bash
$ dream "cosmic explosion" --center-bias
```

### Output Directory

You can specify a directory for saving generated images:

```bash
$ dream "starry mountain landscape" --output_dir=my_images
```

The directory will be created if it doesn't exist.

### Debug Mode

If you encounter issues, you can enable debug mode to see detailed information:

```bash
$ dream "cosmic landscape" --debug
```

This will print additional information about the generation process and file operations.

## Advanced

You can invoke this in code with

```python
from big_sleep import Imagine

dream = Imagine(
    text = "fire in the sky",
    lr = 5e-2,
    save_every = 25,
    save_progress = True
)

dream()
```

> You can now train more than one phrase using the delimiter "|"

### Train on Multiple Phrases
In this example we train on three phrases:

- `an armchair in the form of pikachu` 
- `an armchair imitating pikachu`
- `abstract`

```python
from big_sleep import Imagine

dream = Imagine(
    text = "an armchair in the form of pikachu|an armchair imitating pikachu|abstract",
    lr = 5e-2,
    save_every = 25,
    save_progress = True
)

dream()
```

### Penalize certain prompts as well!

In this example we train on the three phrases from before,

**and** *penalize* the phrases:
- `blur`
- `zoom`
```python
from big_sleep import Imagine

dream = Imagine(
    text = "an armchair in the form of pikachu|an armchair imitating pikachu|abstract",
    text_min = "blur|zoom",
)
dream()
```


You can also set a new text by using the `.set_text(<str>)` command

```python
dream.set_text("a quiet pond underneath the midnight moon")
```

And reset the latents with `.reset()`

```python
dream.reset()
```

To save the progression of images during training, you simply have to supply the `--save-progress` flag

```bash
$ dream "a bowl of apples next to the fireplace" --save-progress --save-every 100
```

Due to the class conditioned nature of the GAN, Big Sleep often steers off the manifold into noise. By default, the program now saves the best high scoring image (per CLIP critic) to `{filepath}.best.png` in your folder. This is often a better result than the final image.

If you don't want this behavior, you can disable it:

```bash
$ dream "a room with a view of the ocean" --save-best=False
```

## Larger model

If you have enough memory, you can also try using a bigger vision model released by OpenAI for improved generations.

```bash
$ dream "storm clouds rolling in over a white barnyard" --larger-model
```

## Experimentation

You can set the number of classes that you wish to restrict Big Sleep to use for the Big GAN with the `--max-classes` flag as follows (ex. 15 classes). This may lead to extra stability during training, at the cost of lost expressivity.

```bash
$ dream 'a single flower in a withered field' --max-classes 15
```

## Alternatives

<a href="https://github.com/lucidrains/deep-daze">Deep Daze</a> - CLIP and a deep SIREN network

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{brock2019large,
    title   = {Large Scale GAN Training for High Fidelity Natural Image Synthesis}, 
    author  = {Andrew Brock and Jeff Donahue and Karen Simonyan},
    year    = {2019},
    eprint  = {1809.11096},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
