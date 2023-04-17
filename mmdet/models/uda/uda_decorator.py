from copy import deepcopy
from mmcv.parallel import MMDistributedDataParallel
from mmdet.models import BaseDetector, build_detector


def get_module(module):
    if isinstance(module, MMDistributedDataParallel):
        return module.module
    return module


class UDADecorator(BaseDetector):

    def __init__(self, **cfg):
        super(BaseDetector, self).__init__()

        self.model = build_detector(deepcopy(cfg['model']))
        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']

    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img):
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        return self.get_model().encode_decode(img, img_metas)

    def forward_train(self, img, img_metas, gt_semantic_seg,  target_img, target_img_metas, return_feat=False):
        losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=return_feat)
        return losses

    def inference(self, img, img_meta, rescale):
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        return self.get_model().aug_test(imgs, img_metas, rescale)
