
import torch
import torch.nn.functional as F
from model import Model
from PIL import Image
import time
import json

torch.cuda.set_device(0)
classes=2
backbone="ResNet50FPN"
model_path='retinanet_rn50fpn.pth'
image_path='anquanmao1.jpg'
detections_file='out.json'
model = Model(backbone, classes)
model, state = Model.load(model_path)
model=model.cuda()
model.eval()
stride = model.stride
print(stride)
im = Image.open(image_path).convert("RGB")
resize = 800
max_size=1333
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
ratio = resize / min(im.size)
if ratio * max(im.size) > max_size:
    ratio = max_size / max(im.size)
im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)
# Convert to tensor and normalize
data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
data = data.float().div(255).view(*im.size[::-1], len(im.mode))
data = data.permute(2, 0, 1)

for t, mean, std in zip(data, mean, std):
    t.sub_(mean).div_(std)

# Apply padding
pw, ph = ((stride - d % stride) % stride for d in im.size)
print(pw,ph)
data = F.pad(data, (0, pw, 0, ph))
data.unsqueeze_(0)
data=data.cuda()
start=time.time()
scores, boxes, classes = model(data)
print(time.time()-start)
#print(scores,boxes,classes)

# Collect detections
detections = []
processed_ids = set()
#for scores, boxes, classes, image_id, ratios in zip(*results):
if True:
    scores, boxes, classes, image_id, ratios = scores[0].cpu(), boxes[0].cpu(), classes[0].cpu(), 0, ratio
    processed_ids.add(image_id)

    keep = (scores > 0).nonzero()
    scores = scores[keep].view(-1)
    boxes = boxes[keep, :].view(-1, 4) / ratios
    classes = classes[keep].view(-1).int()

    for score, box, cat in zip(scores, boxes, classes):
        x1, y1, x2, y2 = box.data.tolist()
        cat = cat.item()
        detections.append({
            'image_id': image_id,
            'score': score.item(),
            'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            'category_id': cat
        })

if detections:
    # Save detections
    detections = { 'annotations': detections }
    if detections_file:
        json.dump(detections, open(detections_file, 'w'), indent=4)
        print('saved to json')
