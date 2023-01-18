from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

class Detector:
    def __init__(self, mode = "zoo", model = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", device = "cpu") -> None:
        self.cfg = get_cfg()

        if mode == "zoo":
            # Load model config and pretrained model
            self.cfg.merge_from_file(model_zoo.get_config_file(model))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(self.cfg)

    def image(self, image):
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        #instance_mode = ColorMode.IMAGE_BW

        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        result = []
        for i in range(len(predictions["instances"].scores)):
            result.append(
                {
                    "box": tuple([float(pos) for pos in [obj for obj in predictions["instances"].pred_boxes[i]][0]]),
                    "score": float(predictions["instances"].scores[i]),
                    "class": MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[predictions["instances"].pred_classes[i]]
                }
            )

        return output.get_image()[:,:,::-1], tuple(result)