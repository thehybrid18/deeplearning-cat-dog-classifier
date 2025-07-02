# 🐾 Cat vs Dog Classifier - ResNet50 + SageMaker

![fireword](https://github.com/user-attachments/assets/228a3108-a534-4927-b7cb-b5067e60ca1a)


A fun retro-themed image classification app that uses deep learning to distinguish between cats and dogs! This project demonstrates how to fine-tune a pretrained ResNet50 model, optimize it with Amazon SageMaker, and deploy it as a lightweight web app.

## 🚀 Project Summary

This project shows the end-to-end ML pipeline:
- Preprocessing and splitting a balanced dataset
- Fine-tuning a ResNet50 model via transfer learning
- Hyperparameter tuning on AWS SageMaker
- Exporting and using the `.pth` model in a Flask web app

It mirrors real-world use cases in medical imaging, object detection, and smart security — though this one is funny and uses pets 🐶🐱.

## 🧠 Model Architecture

I used a pretrained `ResNet50` model from PyTorch. The final layer is replaced with:

```python
nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)
```
The baseline model around 70 percent, while not poor, left significant room for improvement. In production environments, where accuracy and reliability are key (e.g., veterinary clinics or medical imaging platforms), models generally need to exceed 90% accuracy. A 67.3% result is a good starting point, but higher accuracy is strongly preferred to ensure trustworthy predictions. I was able to get 99 percent with a few tuning jobs.

## 📊 Performance

| Metric      | Baseline | Tuned Model |
|-------------|----------|-------------|
| Accuracy    | 67.3%    | 98.9%       |
| Test Loss   | 0.32     | 0.03        |

SageMaker tuning targeted `Test Loss` using 6 parallel jobs varying `learning_rate` and `batch_size` (hyperparameter tuning).

I was impressed with the results—after just a few tuning jobs and no exhaustive architecture tweaks, I achieved ~99% accuracy. With more time, improvements like `Dropout`, `BatchNorm`, or data augmentation (e.g., color jitter, rotation) could help further regularize performance.

![best tuned job parameters](https://github.com/user-attachments/assets/01226f81-1896-438d-b44f-ecb882c92857)


## 🧪 Dataset

Used the [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset:
- 25,000 images (12,500 cats / 12,500 dogs)
- Split: 60% train / 20% validation / 20% test
- Images resized to 224x224

## 🛠️ Tech Stack

- **Model Training**: PyTorch + AWS SageMaker
- **Hyperparameter Tuning**: SageMaker `HyperparameterTuner`
- **Web Framework**: Flask + HTML/CSS
- **Deployment Options**: EC2 or Render
- **Security Considerations**: HTTPS, file upload limits, logging, Authentication/Authorization, and basic sanitization

![diagram1](https://github.com/user-attachments/assets/8f48e332-eb87-4faa-b9b9-339d69c2c486)


## 🖼️ Web Demo

The final website is retro-styled with animated flames and spinning cats. It allows a user to upload an image and get a prediction: **cat** or **dog**.

> “This site has the best graphics I’ve ever seen.”  
> — *Roger Ebert, probably* 😂

![Capture](https://github.com/user-attachments/assets/dc1a8693-6d03-4fb4-9eab-12df49b42a4b)


## 📂 Files

- `app.py`: Flask app script
- `model.pth`: Trained ResNet50 model
- `/templates/index.html`: UI layout
- `/static/`: CSS, gifs, favicon, background images

[test it locally with flask- files are on GOOGLE DRIVE](https://drive.google.com/file/d/1jxdOWrY-LEY68Knk2IRV-uuq4vfW0Z7S/view?usp=sharing)

## 🧪 How to Run Locally

```bash

cd catdog_web_app
pip install -r requirements.txt
python app.py
```

![flasktest](https://github.com/user-attachments/assets/8e14b311-9904-4fea-a40e-2b92cba1e127)



## 📽️ Bonus

🎬 30-second trailer coming soon. VHS-style. 5-star rated. Probably.
[DEMO TRAILER](https://drive.google.com/file/d/1oenEl_68B4zmhlkCY8DA3vD6ctPbWIZe/view?usp=drive_link)

💾 Possibly my greatest work—if it were 1997 🤣🤣

## 📚 References

1. [ResNet50 Paper](https://arxiv.org/abs/1512.03385)  
2. [Amazon SageMaker Docs](https://docs.aws.amazon.com/sagemaker/)  
3. [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats)  
4. [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
