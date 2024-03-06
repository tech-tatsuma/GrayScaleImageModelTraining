# GrayScaleImageModelTraining
This repository is designed for training models including Vision Transformer, MLP-Mixer, ResNet50, and CrossViT on grayscale images with single-channel input. For execution tests, we are utilizing ImageNet.
## How to Use
1. First, please create a dataset with the following structure:
```
Class1/
├── image1
├── image2
├── ...
Class2/
├── image1
├── image2
├── ...
...
ClassN/
├── image1
├── image2
├── ...

```
For the purposes of this explanation, assume the dataset is prepared in 'sample_dataset/'.
2. Modify the program in the train file to fit the desired execution conditions as follows:
```
args = argparse.Namespace(
        epochs=100,
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        patience=20,
        seed=42,
        batch_size=20,
        modelname={MLP-Mixer or ResNet50 or VisionTransformer or CrossViT},
        output_dir={output directory such as ./}
    )
```
3. 
## Networks Available for Validation
- MLP-Mixer
- Vision Transformer
- Resnet 50
- CrossViT