# AGSSP
Official PyTorch Implementation of [Advancing Metallic Surface Defect Detection via Anomaly-Guided Pretraining on a Large Industrial Dataset](#)

Pretrained models are commonly employed to improve finetuning performance in metallic surface defect detection, especially in data-scarce environments. However, models pretrained on ImageNet often underperform due to data distribution gaps and misaligned training objectives. To address this, we propose a novel method called Anomaly-Guided Self-Supervised Pretraining (AGSSP), which pretrains on a large industrial dataset containing 120,000 images. AGSSP adopts a two-stage framework: (1) anomaly map guided backbone pretraining, which integrates domain-specific knowledge into feature learning through anomaly maps, and (2) anomaly box guided detector pretraining, where pseudo-defect boxes derived from anomaly maps act as targets to guide detector training. Anomaly maps are generated using a knowledge enhanced anomaly detection method. Additionally, we present two small-scale, pixel-level labeled metallic surface defect datasets for validation. Extensive experiments demonstrate that AGSSP consistently enhances performance across various settings, achieving up to a 10\% improvement in mAP@0.5 and 11.4% in mAP@0.5:0.95 compared to ImageNet-based models.

![](./images/main.png)


## Data Download

Casting Billet and Steel Pipe can be downloaded from https://github.com/clovermini/MVIT_metal_datasets.


## Citation
If you find our work is useful in your research or applications, please consider giving us a star 🌟 and citing it.
