from ctypes import *
import math
import random

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

# add draw_detections muzhi
get_labels = lib.get_labels
get_labels.argtypes = [c_char_p]
get_labels.restype = POINTER(POINTER(c_char))

load_alphabet = lib.load_alphabet
load_alphabet.restype = POINTER(POINTER(IMAGE))

draw_detections = lib.draw_detections
draw_detections.argtypes = [IMAGE, POINTER(DETECTION), c_int, c_float, POINTER(POINTER(c_char)), POINTER(POINTER(IMAGE)), c_int]

save_image = lib.save_image
save_image.argtypes = [IMAGE, c_char_p]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def detect_and_draw(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, name_list="data/coco.names"):
    names = get_labels(name_list)
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    alphabet = load_alphabet()
    if nms:
        do_nms_obj(dets, num, meta.classes, nms)
    draw_detections(im, dets, num, thresh, names, alphabet, meta.classes)
    # res = []
    # for j in range(num):
    #     for i in range(meta.classes):
    #         if dets[j].prob[i] > 0:
    #             b = dets[j].bbox
    #             res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    # res = sorted(res, key=lambda x: -x[1])
    save_image(im, "predictions")
    free_image(im)
    free_detections(dets, num)

def draw_detected(img, coords):
    for co in coords:
        coord = co[2]
        left  = (coord[0]-coord[2]/2.)
        right = (coord[0]+coord[2]/2.)
        top   = (coord[1]-coord[3]/2.)
        bot   = (coord[1]+coord[3]/2.)

        if left < 0:
            left = 0
        if right > img.shape[1]-1:
            right = img.shape[1]-1
        if top < 0:
            top = 0
        if bot > img.shape[0]-1:
            bot = img.shape[0]-1
        print left, right, top, bot
        draw_img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bot)), (255, 0, 0), 1)
    return draw_img

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
    meta = load_meta("cfg/coco.data")
    # r = detect(net, meta, "data/person.jpg")
    detect_and_draw(net, meta, "data/person.jpg")
    # img = cv2.imread("data/person.jpg", 1)
    # print img.shape
    # r = draw_detected(img, r)
    # cv2.imwrite('image.jpg', r)
    # print r

