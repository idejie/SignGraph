# SignGraph: A Sign Sequence is Worth Graphs of Nodes
An implementation of the paper: SignGraph: A Sign Sequence is Worth Graphs of Nodes. (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Gan_SignGraph_A_Sign_Sequence_is_Worth_Graphs_of_Nodes_CVPR_2024_paper.pdf)

## Prerequisites

```bash
conda create -n sign python=3.12
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 torchtext -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
git clone --recursive https://github.com/WayenVan/ctcdecode.git
cd ctcdecode&& pip install -e .
cd.. && pip install -r requirements.txt
```


## Data Preparation
 
1. PHOENIX2014 dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). 

2. PHOENIX2014-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

3. CSL dataset： Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

Download datasets and extract them to `data`, no further data preprocessing needed. 

## Evaluation

​To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.
 
### Thanks

This repo is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html), [VIT (NIPS 2022)](https://arxiv.org/abs/2206.00272) and [RTG-Net (ACM MM2023)](https://dl.acm.org/doi/10.1145/3581783.3611820)！

### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@inproceedings{gan2024signgraph,
  title={SignGraph: A Sign Sequence is Worth Graphs of Nodes},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Wen, Hongkai and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13470--13479},
  year={2024}
}

@inproceedings{gan2023towards,
  title={Towards Real-Time Sign Language Recognition and Translation on Edge Devices},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4502--4512},
  year={2023}
}

@inproceedings{gan2023contrastive,
  title={Contrastive learning for sign language recognition and translation},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xia, Kang and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23},
  pages={763--772},
  year={2023}
}

@article{han2022vision,
  title={Vision gnn: An image is worth graph of nodes},
  author={Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={8291--8303},
  year={2022}
} 
```
