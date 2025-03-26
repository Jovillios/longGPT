# LongGPT

A project focused on training and experimenting with long-context language models, based on the nanoGPT architecture, for the [LLM MVA course](https://github.com/nathanael-fijalkow/llm_mva) by Nathanael Fijalkow

## Author

- Jules Decaestecker
- Ibrahim Ridene

## Overview

This project extends the nanoGPT architecture to explore and experiment with long-context language modeling. It provides a simple and efficient implementation for training and fine-tuning language models with extended context windows.

## Features

- Efficient implementation of transformer-based language models
- Support for long-context training and inference
- Easy-to-use training and sampling scripts
- Configurable model architecture and training parameters
- Integration with popular datasets and tokenizers

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- PyTorch
- NumPy
- Transformers 
- Datasets
- Tiktoken
- Weights & Biases 
- tqdm
- einops
- xformers
- Requests

## Project Structure

- `model.py`: Core model implementation
- `train.py`: Training script
- `sample.py`: Text generation script
- `config/`: Configuration files for different training scenarios
- `data/`: Data processing and dataset management
- `tests/`: Test suite
- `assets/`: Project assets and resources

## Usage

### Training

To train a model, use the training script with a configuration file:

```bash
python train.py config/train_config.py
```

### Sampling

To generate text from a trained model:

```bash
python sample.py --out_dir=path/to/model/checkpoint
```

## Development

This project is under active development. Contributions are welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
