import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np

from SFD.bbox import nms, decode

torch.backends.cudnn.bencmark = True


def detect(net, img):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)

    img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
    BB, CC, HH, WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist) // 2): olist[i * 2] = F.softmax(olist[i*2], 1)
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2].data.cpu(), olist[i * 2 + 1].data.cpu()
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2 ** (i + 2)  # 4,8,16,32,64,128
        anchor = stride * 4
        for Findex in range(FH * FW):
            windex, hindex = Findex % FW, Findex // FW
            axc, ayc = stride/2 + windex * stride, stride/2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            if score < 0.05: continue
            priors = torch.Tensor(
                [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist): bboxlist = np.zeros((1, 5))
    return bboxlist


def output(net, im_path):
    """Output detection results of faces"""

    img = cv2.imread(im_path)
    bboxlist = detect(net, img)

    keep = nms(bboxlist, 0.3)
    bboxlist = bboxlist[keep, :]

    bbox_list = []
    for box in bboxlist:
        x1, y1, x2, y2, score = box
        if score < 0.5:
            continue
        x1, y1, x2, y2 = (int(x) for x in [x1, y1, x2, y2])
        bbox_list.append((x1, y1, x2, y2))

    return bbox_list
