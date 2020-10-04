# from datasets.camvid.CamVid import CamVid, ActiveCamVid
from datasets.cityscapes.Cityscapes import Cityscapes, ActiveCityscapes
from utils.vis import get_label_name_colors
import os

data_cfg = {
    'Cityscapes': {
        'cls': (Cityscapes, ActiveCityscapes),
        'root': '/nfs/xs/Datasets/Segment/Cityscapes',
        'base_size': (2048, 1024),
        'crop_size': (688, 688),
        'num_classes': 19,
        'label_colors': get_label_name_colors(
            csv_path=os.path.join(os.path.dirname(__file__), 'cityscapes/cityscapes19.csv')
        )
    },
    'CamVid': {
        'root': '/nfs2/xs/Datasets/CamVid11',
    }
}
