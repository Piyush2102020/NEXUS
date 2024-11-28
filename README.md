# Nexus Model

## Overview

The **Nexus** model is a transformer-based language model built using PyTorch, designed to generate text from input sequences. It supports customizable generation settings and allows for easy integration and inference using a simple `generate` function.

This repository includes the pre-trained model and code to run inference on your own input data. It also provides setup instructions for installing necessary dependencies.

## Features

- Transformer-based architecture.
- Easy-to-use text generation via a simple `generate` function.
- Pre-trained model available for quick use.
- Configurable generation settings such as number of samples and token length.

## Requirements

Make sure you have the following libraries installed:

- **torch**: For deep learning computations.
- **transformers**: Hugging Face's library for model loading and tokenization.
- **numpy**: For numerical operations.
- **tqdm**: For progress bars in training and inference.

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Setup
unzip the model_weights

### Loading the Pre-Trained Model

The Nexus model can be loaded and used for text generation. Use the following script to generate text from an input sequence:

```python
from nexus import Nexus

# Initialize the model
model = Nexus()

# Generate text from an input string
generated_text = model.generate("Hey there", num_samples=5, max_new_tokens=10)

# Output the generated text
print(generated_text)
```

### Model Generation Function

The `generate` function is designed to take a string input and generate a specified number of output samples with a defined length:

```python
def generate(input_string, num_samples=1, max_new_tokens=50):
    """
    Generate text based on an input string.

    Args:
        input_string (str): Input text to generate from.
        num_samples (int, optional): Number of samples to generate. Defaults to 1.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 50.
    """

```

### Saving the Model

To save the model for future use, you can use the following code:

```python
torch.save(model.state_dict(), 'nexus_model.pth')
```

## Contributions

Feel free to fork the repository and make improvements. Contributions are welcome!
