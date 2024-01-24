运行环境：
python3.8
pytorch 1.10.0
cuda 10.2
MMOCR 1.0.1
mmdet 3.1.0

测试命令：
python tools/test.py configs/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015.py your_weight_path

训练命令：
python tools/train.py configs/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015.py