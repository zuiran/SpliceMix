## SpliceMix: A Cross-scale and Semantic Blending Augmentation Strategy for Multi-label Image Classification

### [Overview]
![](Etc/overview.png "Overview")

### [Results on MS-COCO2014]
![](Etc/results_coco.png "results")

### [Run scripts]
1. The code of our SpliceMix and SpliceMix-CL methods are put to ./SpliceMix.py and ./models/SpliceMix_CL.py, respectively.

2. Please download data sets by yourself and follow the below structure to unzip files. Then modify 'args.data_root' to DATA_ROOT.
```
    Data root structure:
        DATA_ROOT
            --COCO2014
                --train2014
                    --...
                --val2014
                    --...
                --category.json
                --train_anno.json
                --val_anno.json
                --...

            --VOC2007
                --VOCdevkit
                    --VOC2007
                        --JPEGImages
                            --...
                --...
```

3. Run script:
It is recommended to run the code by 'launch.py'.  
a.1) train on MS-COCO
```shell
./launch.sh -m ResNet-101 -mixer SpliceMix--Default=True -ds MS-COCO -lr .05 -wup 3 -bs 32 -cd 0 1 -P 17837 -rmk SpliceMix--Default=True
./launch.sh -m SpliceMix-CL -mixer SpliceMix--Default=True -ds MS-COCO -lr .05 -wup 3 -bs 32 -cd 0 1 -P 17837 -rmk SpliceMix--Default=True
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
a.2) inference on MS-COCO  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
     The pre-training weight of 'ResNet-101+SpliceMix' and 'ResNet-101+SpliceMix-CL' on MS-COCO is available at https://drive.google.com/drive/folders/1VwKrEqAYYE9m7raVMwhyza9Fwjy9slCS?usp=sharing.  
```shell  
./launch.sh -m ResNet-101 -ds MS-COCO -bs 32 -cd 0 -e 0 -P 17837 -r checkpoint/ResNet_101_SpliceMix.pt -rmk SpliceMix
./launch.sh -m ResNet-101 -ds MS-COCO -bs 32 -cd 0 -e 0 -P 17837 -r checkpoint/SpliceMix_CL.pt -rmk SpliceMix  
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
b) train on VOC2007  
```
./launch.sh -m ResNet-101 -mixer SpliceMix--Default=True -ds VOC2007 -lr .01 -bs 16 -cd 0 -P 17837 --disable-amp -rmk SpliceMix--Default=True
./launch.sh -m SpliceMix-CL -mixer SpliceMix--Default=True -ds VOC2007 -lr .01 -bs 16 -cd 0 -P 17837 --disable-amp -rmk SpliceMix--Default=True
```
* -rmk is optional for checkpoint folder suffix.  
* The offered code is a castrated version. There could exist some bugs. Running the given script is fine.
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
c) SpliceMix pre-training on ImageNet  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   The pre-training model using SpliceMix trained on ImageNet is also available at https://drive.google.com/drive/folders/1VwKrEqAYYE9m7raVMwhyza9Fwjy9slCS?usp=sharing. It can be easily loaded to ResNet in PyTorch, such as   
```python
ResNet = torchvision.models.resnet101(pretrained=false)
file = r'checkpoint/ImageNet_ResNet101_SpliceMix_te79.912_E163.pth.tar'
ckpt = torch.load(file, map_location='cpu')
ResNet.load_state_dict(ckpt['state_dict'])
del ckpt
```

### [.bib citation]
cite the paper as follows (copied-pasted it from arxiv for you):
```
@article{wang2023splicemix,
  title={SpliceMix: A Cross-scale and Semantic Blending Augmentation Strategy for Multi-label Image Classification},
  author={Wang, Lei and Zhan, Yibing and Ma, Leilei and Tao, Dapeng and Ding, Liang and Gong, Chen},
  journal={arXiv preprint arXiv:2311.15200},
  year={2023}
}
```
