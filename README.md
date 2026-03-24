# HASNet
The official code for the paper ‘Hybrid aggregation strategy with double inverted residual blocks for lightweight salient object detection’
# 

## Saliency maps 
[Google Drive](https://drive.google.com/drive/folders/1E7rC8N_vNplL_-T7G6RrQYTPkehoTsQH?usp=sharing)






### If you think this work is helpful, please cite

```latex
@article{MA2026108097,
title = {Hybrid aggregation strategy with double inverted residual blocks for lightweight salient object detection},
journal = {Neural Networks},
volume = {194},
pages = {108097},
year = {2026},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2025.108097},
url = {https://www.sciencedirect.com/science/article/pii/S0893608025009773},
author = {Jianhua Ma and Mingfeng Jiang and Xian Fang and Jiatong Chen and Yaming Wang and Guang Yang},
keywords = {Lightweight salient object detection, Inverted residual bolck, Feature aggregation strategy, Hybrid encoder},
abstract = {Lightweight salient object detection (SOD) is widely used in various downstream applications due to its low resource requirements and fast inference speed. The use of hybrid encoders offers the potential to achieve a better balance between efficiency and accuracy for SOD task. However, the aggregation of features from convolutional neural networks (CNNs) and transformers remains challenging, and most existing lightweight SOD models rarely explore the efficient aggregation of cross-architecture features derived from hybrid encoders. In this paper, we propose a hybrid aggregation strategy network (HASNet) that balances accuracy and efficiency for lightweight SOD by grouping and aggregating features to leverage salient information across different architectures. Specifically, the features obtained after hybrid encoder processing are divided into convolutional and transformer features for shallow and deep aggregation respectively. Deep aggregation uses the global inverted residual block (GIRB) to facilitate the transfer of salient information encoded within transformer features across various levels. Meanwhile, shallow aggregation uses the lightweight inverted residual block (LIRB) to efficiently integrate the spatial information inherent in convolutional features. The GIRB incorporates an efficient global operation to extract channel semantic information from the high-dimensional transformer features. The LIRB fuses low-level features by efficiently exploiting the spatial information in features at extremely low computational cost. Comprehensive experiments conducted across five datasets demonstrate that our HASNet significantly outperform existing methods in a thorough evaluation encompassing parameter sizes, inference speed, and accuracy. The source code will be publicly available at https://github.com/LitterMa-820/HASNet.}
}
```
