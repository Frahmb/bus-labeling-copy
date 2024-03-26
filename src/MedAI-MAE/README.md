## MedAI Modifications to the Masked Autoencoders: A PyTorch Implementation for Binary Classifications

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This repository holds source code from Meta AI PyTorch implementation of a masked autoencoder (MAE) machine learning model. All rights are reserved to the righftul owners Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. The original repository can be found [here](https://github.com/facebookresearch/mae). Please look at the original repository's README document before reading this one. The paper the model was implemented on can be found [here](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```
This project focuses on finetuning a pretrained MAE model available from Meta AI open source code on the task of early breast cancer detection. This is a binary classification task (malignant and benign). Necessary modifications and additions are made to the model architecture to allow for binary classification. 

### Installation
Clone this repository. PyTorch and GPU are needed. 

### Checkpoints
We used the pretrained ViT-Base model checkpoints to fine-tune the model. The pretrained ViT-Base model checkpoints can be downloaded [here] (https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth).

### Datasets
Split labeled dataset into train and val folders. In order to run the model, the data needs to be in a folder following the format: ~\data\train\label1\img1.png and ~\data\val\label1\img1.png.

### Modifications and Additions
#### Model Architecture
The original model was pretrained using imagenet images for 1000 classes. We had to change the final layer architecture from 1000 classes to 2 classes. This modification was made in main_finetune.py on lines 240-246. 

#### Loss Function
Cross entropy loss was used. This decision came after failing to modify the code to adapt to binary cross entropy loss. 

#### Evaluation
The original evaluation used a top1 and top5 accuracy. Since this task is for binary classification, the top5 accuracy is not relevant. We changed the top5 to a top2 accuracy to get rid of errors. The top2 accuracy is always 100% because there are only two classes, so it is not necessary but was kept in the code to avoid errors. In addition to accuracy, sensitivity and specificity was calculated using a confusion matrix imported from sklearn.metrics. These modifications can be found in engine_finetune.py in the evaluate() function. In addition, the testing output was modified to print the sensitivity and specificity. 

### Fine-tuning
To finetune this model, a GPU is needed. Finetuning is accomplished using the command below. Further detail on the arguments can be found in the original Meta AI repository's PRETRAIN.md found [here](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md). 

```
python main_finetune.py --finetune mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGE_DIR} --nb_classes 2
```

### Results
After fine-tuning, the model had a 97.6% accuracy. After evaluation on a test set of 815 images, the model produced an accuracy of 98.16%, sensitivity of 98.1%, and specificity of 98.2%. 

### License (from original repository)
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
