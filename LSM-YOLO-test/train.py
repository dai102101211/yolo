import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/LSM-YOLO/LSM-YOLO.yaml')
    model.train(data=r'C:\yolo\LSM-YOLO\data.yaml',
                cache=False,
                project='runs/train',
                name='exp4',
                epochs=1,
                batch=48,
                close_mosaic=0,
                optimizer='SGD', # using SGD
                device='',
                # resume='', # last.pt path
                )