# JurasSigLIP

![Text-guided segmentation example](cover.jpg)

JurasSigLIP is a text-guided segmentation prototype that fuses DINOv3 vision tokens with SigLIP2 text tokens via cross-attention. It aligns global image and text features with a contrastive objective, then trains a segmentation head to produce pixel-level masks conditioned on natural-language captions.

## Project highlights

- Vision backbone: DINOv3 ViT (LoRA-adapted), patch tokens and CLS token
- Text backbone: SigLIP2 text tower (LoRA-adapted), sequence tokens and global text feature
- Fusion: multi-head cross-attention from vision patches to text tokens
- Outputs: segmentation mask logits upsampled to 512x512
- Training: contrastive pretraining for global alignment, then segmentation finetuning with Dice + Focal loss

## Architecture

```mermaid
graph TD
    subgraph Inputs
    IMG[Input Image] --> V_PROC[Image Processor]
    TXT[Input Caption] --> T_TOK[Tokenizer]
    end

    subgraph Vision_Branch_DINOv3
    V_PROC --> V_BB["DINOv3 Backbone<br/>(LoRA Adapted)"]
    V_BB --> V_TOK["Patch Tokens<br/>(Local Features)"]
    V_BB --> V_CLS["CLS Token<br/>(Global Feature)"]
    end

    subgraph Text_Branch_SigLIP2
    T_TOK --> T_BB["SigLIP2 Text Model<br/>(LoRA Adapted)"]
    T_BB --> T_FEAT[Text Features]
    T_FEAT --> T_PROJ[Down Projection]
    T_PROJ --> T_SEQ[Text Sequence Tokens]
    T_PROJ --> T_GLO[Global Text Feature]
    end

    subgraph Fusion_Module
    V_TOK -- Query --> X_ATTN[Cross Attention]
    T_SEQ -- Key/Value --> X_ATTN
    X_ATTN --> V_CTX[Contextualized Features]
    V_TOK -- Residual --> ADD((+))
    V_CTX --> ADD
    end

    subgraph Outputs
    ADD --> SEG_HEAD["Segmentation Head<br/>(MLP + Upsample)"]
    SEG_HEAD --> MASK[Segmentation Mask Logits]
    
    V_CLS -.-> CONT_LOSS[Contrastive Loss]
    T_GLO -.-> CONT_LOSS
    end

    style Vision_Branch_DINOv3 fill:#e1f5fe,stroke:#01579b
    style Text_Branch_SigLIP2 fill:#fff3e0,stroke:#e65100
    style Fusion_Module fill:#f3e5f5,stroke:#4a148c
    style Outputs fill:#e8f5e9,stroke:#1b5e20
```

## Data

The notebook uses COCO train2014 images and a GRef-style referring expression dataset (via `grefs(unc).json`) with instance masks from COCO annotations. Each image can have multiple captions, and the dataset pairs each caption with the same segmentation mask for that target.

## Training flow

1. Contrastive pretraining
   - Align global image and text embeddings with a CLIP-style contrastive loss.
   - Train LoRA-adapted backbones plus logit scale.
2. Segmentation training
   - Freeze most parameters; train cross-attention and segmentation head.
   - Optimize Dice + Focal loss on per-caption masks.

Key hyperparameters from the notebook:
- Shared embedding dim: 384
- Image size: 512
- LoRA: r=8, alpha=16, dropout=0.0
- Backbones: DINOv3 ViT-S/16+, SigLIP2 base patch16-512

## Training curves

Contrastive pretraining:

![Train contrastive loss](curves/train_con_loss.png)
![Val contrastive loss](curves/val_con_loss.png)

Segmentation finetuning:

![Train segmentation loss](curves/train_seg_loss.png)
![Val segmentation loss](curves/val_seg_loss.png)

## Validation comparisons

Each folder in `val/` contains one image with multiple captions. The different captions produce different segmentation outputs.

### Example 1

![example1_1](val/eg1/val_0004_58.jpg)
![example1_2](val/eg1/val_0008_54.jpg)
![example1_3](val/eg1/val_0009_33.jpg)

### Example 2

![example2_1](val/eg2/val_0007_13.jpg)
![example2_2](val/eg2/val_0033_10.jpg)
![example2_3](val/eg2/val_0049_40.jpg)

### Example 3

![example3_1](val/eg3/val_0008_02.jpg)
![example3_2](val/eg3/val_0010_58.jpg)
![example3_3](val/eg3/val_0034_40.jpg)

### Example 4

![example4_1](val/eg4/val_0015_26.jpg)
![example4_2](val/eg4/val_0020_32.jpg)

### Example 5

![example5_1](val/eg5/val_0020_06.jpg)
![example5_2](val/eg5/val_0026_09.jpg)
![example5_3](val/eg5/val_0027_28.jpg)

### Example 6

![example6_1](val/eg6/val_0025_04.jpg)
![example6_2](val/eg6/val_0055_35.jpg)
![example6_3](val/eg6/val_0057_36.jpg)

### Example 7

![example7_1](val/eg7/val_0028_39.jpg)
![example7_2](val/eg7/val_0043_48.jpg)

### Example 8

![example8_1](val/eg8/val_0028_41.jpg)
![example8_2](val/eg8/val_0033_17.jpg)

### Example 9

![example9_1](val/eg9/val_0029_25.jpg)
![example9_2](val/eg9/val_0056_02.jpg)

### Example 10

![example10_1](val/eg10/val_0030_38.jpg)
![example10_2](val/eg10/val_0043_08.jpg)
![example10_3](val/eg10/val_0043_36.jpg)

## How to run

Open `jurasSigLIP.ipynb` and run the cells in order. Update the dataset paths to your local setup and ensure a CUDA-capable GPU is available for mixed-precision training. You must provide the two JSON files (`grefs(unc).json` and `instances.json`) and a folder with all segmented images named `gref_images` as referenced in the notebook paths.

## References

- Liu, Chang, Henghui Ding, and Xudong Jiang. “GRES: Generalized Referring Expression Segmentation.” In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- He, Shuting, Henghui Ding, Chang Liu, and Xudong Jiang. “GREC: Generalized Referring Expression Comprehension.” arXiv preprint arXiv:2308.16182, 2023.
- Kamath, Aishwarya, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. “MDETR: Modulated Detection for End-to-End Multi-Modal Understanding.” arXiv preprint arXiv:2104.12763, 2021.
