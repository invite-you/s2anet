import cv2
import json
import numpy as np

from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoDotaOBBOnlyCARDataset(CustomDataset):
    CLASSES = ('small car', 'bus', 'truck', 'train')

    def load_annotations(self, ann_file):
        '''
        load annotations from .json ann_file
        '''
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.img_ids = self.coco.getImgIds()
        self.img_infos = []
        self.img_names = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            self.img_infos.append(info)
            self.img_names.append(info['file_name'])
        return self.img_infos

    def get_ann_info(self, idx):
        ann = {'bboxes': [],
                'bboxes_ignore': None,
                'labels': [],
                'labels_ignore': None}

        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns_info = self.coco.loadAnns(ann_ids)

        for ann_info in anns_info:    
            coord = np.array([ [x, y] for x, y in zip(ann_info['segmentation'][0][::2], ann_info['segmentation'][0][1::2])] , dtype=np.float32)
            rbox = cv2.minAreaRect(coord)
            ann['bboxes'].append( [rbox[0][0], rbox[0][1], rbox[1][0] , rbox[1][1], rbox[2]] )
            ann['labels'].append( self.cat2label[ann_info['category_id']] )
                    

        ann['bboxes'] = np.array(ann['bboxes'], dtype=np.float32) 
        ann['bboxes_ignore'] = np.array([])
        ann['labels'] = np.array(ann['labels'], dtype=np.int64)
        ann['labels_ignore'] = np.array([])

        return ann

    def get_cat_ids(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns_info = self.coco.loadAnns(ann_ids)
        
        return [self.cat2label[ann_info['category_id']] for ann_info in anns_info]
