# Damper Defect Detection

### Spectrogram-based Classification of Time Series Using Computer Vision Techniques

In automotive technology, chassis system components such as dampers have great importance in ensuring the driving safety and good stability of vehicles. Thus, it is crucial to regularly check, test, and maintain components to detect any issues and prevent potential malfunctions. This project implements Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for identifying damper defects in vehicles. Presently, this task relies on the driver's observations, even though there exist model-based approaches in the literature. With the increase of data available in modern vehicles and the recent developments in computer vision, this paper explores the potential of CNNs and ViTs for diagnosing automotive damper defects using the dataset including 100,000 km of real-world driving comprising of accelerometer signals of the vehicle body.

# Usage
### Data Generation


    python data_generation.py
### Training
To train ViT:

    python train_vit.py
To train CNN:

    python train_cnn.py

### Activation Map Generation


    python attention_map_vit.py
    python class_activation_map_cnn.py