# End-to-end Deep Multi-Score Model for No-reference Stereoscopic Image Quality Assessment

## Overview

This repository contains the implementation of an end-to-end deep multi-score model for no-reference stereoscopic image quality assessment (SIQA). The model predicts multiple quality scores, including global quality, left view quality, right view quality, and stereo quality, without requiring a reference image.

## Features

- **Multi-Score Prediction**: Predicts global, left, right, and stereo quality scores.
- **No-reference Assessment**: Evaluates image quality without the need for a reference image.
- **Browser-based Application**: Run the model directly in your browser using the online web application.
- **Support for Various Image Formats**: Compatible with BMP, PNG, JPG, GIF, and more.

## Paper

This implementation is based on our paper published at the **2022 IEEE International Conference on Image Processing (ICIP 2022)**: [Read the Paper on arXiv](https://arxiv.org/abs/2211.01374)

## Citation
```bibtex
@proceeding{messai2022-end-to-end,
    title={End-to-end deep multi-score model for No-reference stereoscopic image quality assessment},
    author={Messai, Oussama and Chetouani, Aladine},
    conference={ICIP2022},
    year={2022},
    publisher={IEEE}
}
```

## Online Web Application

Experience the model in action through our online web application:

[https://oussama-messai.com/iqa-stereo](https://oussama-messai.com/iqa-stereo)

**Features:**

- **Edge Mode Execution**: The model runs directly in the browser, leveraging your machine's computational power.
- **Flexible Resolution Support**: Works with any image resolution greater than 32x32 pixels.
- **Data Privacy**: The application does not record or save any user data or images.
- **Feedback Welcome**: While we do not store data, your feedback is greatly appreciated to improve the application.

![Online Web Application](https://github.com/o-messai/multi-score-SIQA/blob/main/results/online.png?raw=true)

## Requirements

Ensure you have the following dependencies installed:

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/)
- [TensorBoardX](https://github.com/lanpa/tensorboardX)
- [torchsummary](https://github.com/sksq96/pytorch-summary)
- SciPy
- NumPy
- Pillow
- Matplotlib
- YAML

You can install all the required packages using the provided `requirements.txt` file.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/o-messai/multi-score-SIQA.git
   cd multi-score-SIQA
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**

   - Place your datasets in the `./data/Waterloo_1/` and `./data/Waterloo_2/` directories.
   - Ensure the following files are present in each dataset directory:
     - `im_names_S.txt`
     - `refnames_S.txt`
     - `ref_ids_S.txt`
     - `MOS_S.txt`
     - `MOS_L.txt`
     - `MOS_R.txt`

## Usage

### Training the Model

To train the model, use the `train.py` script with the desired configuration:

```bash
python train.py --batch_size 128 --epochs 100 --lr 0.0001 --dataset Waterloo_1 --weight_decay 0.001
```

**Parameters:**

    - `--batch_size`: Number of samples per batch (default: 128)
    - `--epochs`: Number of training epochs (default: 100)
    - `--lr`: Learning rate (default: 0.0001)
    - `--dataset`: Dataset to use (`Waterloo_1` or `Waterloo_2`)
    - `--weight_decay`: Weight decay (default: 0.001)

After training, the model will automatically evaluate on the test set and save the performance metrics in `results/total_result.txt`. 

*(Ensure that the evaluation flag and corresponding logic are implemented in `train.py` if not already present.)*

### Testing the Model

To test the model, use the `test.py` script with the desired configuration:

```bash
python test.py --dataset Waterloo_1 --model_path path/to/your/model.pth
```


### Using the Online Web Application

Access the online web application to assess image quality without any setup:

[https://oussama-messai.com/iqa-stereo](https://oussama-messai.com/iqa-stereo)




## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

