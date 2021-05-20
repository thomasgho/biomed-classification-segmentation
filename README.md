# Prostate Ultrasound Classification-Segmentation

A supervised 2D image segmentation tool for transrectal B-mode ultrasound images of prostate gland capsules. Refer here.

Due to the inherent difficulty of identifying capsules, a supervised DenseNet is used to learn detailed feature-maps to classify whether a given image frame contains prostate or not. A supervised UNet is then trained to learn the visual and positional feature-maps characterising prostate gland images to their corresponding labels.

