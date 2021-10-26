import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import RQD


def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load('model_20.pth'))
    model.eval()
    RQD.rqd(model)


if __name__ == '__main__':
    main()
