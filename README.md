# MMD-ReID
Pytorch implementation for MMD-ReID: A Simple but Effective solution for Visible-Thermal Person ReID. Accepted at BMVC 2021 (Oral)

**Paper link**: https://arxiv.org/abs/2111.05059

**Github Code**: https://github.com/vcl-iisc/MMD-ReID

**Presentation Slides**: https://drive.google.com/file/d/1S0sfA7PMyzqGPnG5izGBeZ7uClsJ1uA3/view?usp=sharing

**Project webpage**: https://vcl-iisc.github.io/mmd-reid-web/

**Recorded Talk**: https://recorder-v3.slideslive.com/?share=55344&s=d3b53e98-4362-410a-825d-77706f8b71c4


### Dependencies:

### How to use this code: 
Our code extends the pytorch implementation of [Parameter Sharing Exploration and Hetero center triplet loss for VT Re-ID](https://github.com/hijune6/Hetero-center-triplet-loss-for-VT-Re-ID) in Github. Please refer to the offical repo for details of data preparation.

### Training: 
```
python train_mine.py --dataset sysu --gpu 1 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --dist_disc 'margin_mmd' --margin_mmd 1.40 --run_name 'margin_mmd1.40'
```

### Testing:
```
python test.py --dataset sysu --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --run_name 'margin_mmd1.40'
```

### Results:

### Citation
If you use this code, please cite our work as:
