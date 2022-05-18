from models import yolov3
import cv2


class Detection:
    def __init__(self, weight_directory):
        self.net_h, self.net_w = 416, 416
        self.obj_thresh, self.nms_thresh = 0.5, 0.45
        self.anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

        self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
                  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
                  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
                  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
                  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
                  "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        # make the yolov3 model to predict 80 classes on COCO
        self.model = yolov3.make_yolov3_model()

        # load the weights trained on COCO into the model
        weight_reader = yolov3.WeightReader(weight_directory)
        weight_reader.load_weights(self.model)


    def create_mask(self, image, mask_class):
        image_h, image_w, _ = image.shape

        pre_image = yolov3.preprocess_input(image, self.net_h, self.net_w)

        yolos = self.model.predict(pre_image)

        boxes = []

        for i in range(len(yolos)):
            # decode the output of the network
            boxes += yolov3.decode_netout(yolos[i][0], self.anchors[i], self.obj_thresh, self.nms_thresh, self.net_h,
                                          self.net_w)

        # correct the sizes of the bounding boxes
        yolov3.correct_yolo_boxes(boxes, image_h, image_w, self.net_h, self.net_w)

        mask = yolov3.create_mask(image, boxes, self.labels, self.obj_thresh, mask_class)

        cv2.imwrite("mask.png", mask.astype('uint8'))
        return cv2.imread("mask.png")
