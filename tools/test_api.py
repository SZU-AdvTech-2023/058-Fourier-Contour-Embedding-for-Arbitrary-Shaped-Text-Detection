import matplotlib.pyplot as plt
from mmocr.apis import TextDetInferencer
img = 'data/mini_icdar2015/test/img_444.jpg'
checkpoint = "work_dirs/new_checkpoint.pth"
cfg_file = "configs/textdet/fcenet/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py"
# cfg_file = "work_dirs/fcenet_resnet50_fpn_1500e_icdar2015/fcenet_resnet50_fpn_1500e_icdar2015.py"

infer = TextDetInferencer(cfg_file, checkpoint)
result = infer(img, return_vis=True)

# print(f'result: {result["predictions"]}' )

plt.figure(figsize=(9, 16))
plt.imshow(result['visualization'][0])
plt.show()

# from mmocr.apis import MMOCRInferencer
# import matplotlib.pyplot as plt

# infer = MMOCRInferencer(det='fcenet')
# result = infer(img, return_vis=True)
# plt.figure(figsize=(9, 16))
# plt.imshow(result['visualization'][0])
# plt.show()


# import torch
# # checkpoint = "work_dirs/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500_20220825_221510-4d705392.pth"
# checkpoint = "work_dirs/dcnv2_epoch_1500.pth"
# model = torch.load(checkpoint)

# if 'optimizer' in model:
#     del model['optimizer']

# torch.save(model, 'new_checkpoint.pth')
