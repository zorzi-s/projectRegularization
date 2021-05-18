import random
from skimage import io
from skimage.transform import rotate
import numpy as np
import torch
from tqdm import tqdm
import gdal
import os
import glob
from skimage.segmentation import mark_boundaries
from PIL import Image, ImageDraw, ImageFont
from numpy.linalg import svd
import cv2
from skimage import measure

from models import GeneratorResNet, Encoder
from skimage.transform import rescale
import variables as var




def compute_IoU(mask, pred):
    mask = mask!=0
    pred = pred!=0
    
    m1 = np.logical_and(mask, pred)
    m2 = np.logical_and(np.logical_not(mask), np.logical_not(pred))
    m3 = np.logical_and(mask==0, pred==1)
    m4 = np.logical_and(mask==1, pred==0)
    m5 = np.logical_or(mask, pred)
    
    tp = np.count_nonzero(m1)
    fp = np.count_nonzero(m3)
    fn = np.count_nonzero(m4)
    
    IoU = tp/(tp+(fn+fp)) 
    return IoU


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def predict_building(rgb, mask, model):
	Tensor = torch.cuda.FloatTensor

	mask = to_categorical(mask, 2)

	rgb = rgb[np.newaxis, :, :, :]
	mask = mask[np.newaxis, :, :, :]

	E, G = model

	rgb = Tensor(rgb)
	mask = Tensor(mask)
	rgb = rgb.permute(0,3,1,2)
	mask = mask.permute(0,3,1,2)

	rgb = rgb / 255.0

	# PREDICTION
	pred = G(E([rgb, mask]))
	pred = pred.permute(0,2,3,1)

	pred = pred.detach().cpu().numpy()

	pred = np.argmax(pred[0,:,:,:], axis=-1)
	return pred



def fix_limits(i_min, i_max, j_min, j_max, min_image_size=256):

	def closest_divisible_size(size, factor=4):
		while size % factor:
			size += 1
		return size

	height = i_max - i_min
	width = j_max - j_min

	# pad the rows
	if height < min_image_size:
		diff = min_image_size - height
	else:
		diff = closest_divisible_size(height) - height + 16

	i_min -= (diff // 2)
	i_max += (diff // 2 + diff % 2)

	# pad the columns
	if width < min_image_size:
		diff = min_image_size - width
	else:
		diff = closest_divisible_size(width) - width + 16

	j_min -= (diff // 2)
	j_max += (diff // 2 + diff % 2)

	return i_min, i_max, j_min, j_max



def regularization(rgb, ins_segmentation, model, in_mode="instance", out_mode="instance", min_size=10):
    assert in_mode == "instance" or in_mode == "semantic"
    assert out_mode == "instance" or out_mode == "semantic"

    if in_mode == "semantic":
        ins_segmentation = np.uint16(measure.label(ins_segmentation, background=0))

    max_instance = np.amax(ins_segmentation)
    border = 256

    ins_segmentation = np.uint16(cv2.copyMakeBorder(ins_segmentation,border,border,border,border,cv2.BORDER_CONSTANT,value=0))
    rgb = np.uint8(cv2.copyMakeBorder(rgb,border,border,border,border,cv2.BORDER_CONSTANT,value=(0,0,0)))

    regularization = np.zeros(ins_segmentation.shape, dtype=np.uint16)

    for ins in tqdm(range(1, max_instance+1), desc="Regularization"):
        indices = np.argwhere(ins_segmentation==ins)
        building_size = indices.shape[0]
        if building_size > min_size:
            i_min = np.amin(indices[:,0])
            i_max = np.amax(indices[:,0])
            j_min = np.amin(indices[:,1])
            j_max = np.amax(indices[:,1])

            i_min, i_max, j_min, j_max = fix_limits(i_min, i_max, j_min, j_max)

            mask = np.copy(ins_segmentation[i_min:i_max, j_min:j_max] == ins)
            rgb_mask = np.copy(rgb[i_min:i_max, j_min:j_max, :])



            max_building_size = 1024
            rescaled = False
            if mask.shape[0] > max_building_size and mask.shape[0] >= mask.shape[1]:
                f = max_building_size / mask.shape[0]
                mask = rescale(mask, f, anti_aliasing=False, preserve_range=True)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False)
                rescaled = True
            elif mask.shape[1] > max_building_size and mask.shape[1] >= mask.shape[0]:
                f = max_building_size / mask.shape[1]
                mask = rescale(mask, f, anti_aliasing=False)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False, preserve_range=True)
                rescaled = True

            pred = predict_building(rgb_mask, mask, model)

            if rescaled:
                pred = rescale(pred, 1/f, anti_aliasing=False, preserve_range=True)



            pred_indices = np.argwhere(pred != 0)

            if pred_indices.shape[0] > 0:
                pred_indices[:,0] = pred_indices[:,0] + i_min
                pred_indices[:,1] = pred_indices[:,1] + j_min
                x, y = zip(*pred_indices)
                if out_mode == "semantic":
                    regularization[x,y] = 1
                else:
                    regularization[x,y] = ins

    return regularization[border:-border, border:-border]



def copyGeoreference(inp, output):
    dataset = gdal.Open(inp)
    if dataset is None:
        print('Unable to open', inp, 'for reading')
        sys.exit(1)

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is None and geotransform is None:
        print('No projection or geotransform found on file' + input)
        sys.exit(1)

    dataset2 = gdal.Open(output, gdal.GA_Update)

    if dataset2 is None:
        print('Unable to open', output, 'for writing')
        sys.exit(1)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        dataset2.SetGeoTransform(geotransform)

    if projection is not None and projection != '':
        dataset2.SetProjection(projection)

    gcp_count = dataset.GetGCPCount()
    if gcp_count != 0:
        dataset2.SetGCPs(dataset.GetGCPs(), dataset.GetGCPProjection())

    dataset = None
    dataset2 = None



def regularize_segmentations(img_folder, seg_folder, out_folder, in_mode="semantic", out_mode="instance", samples=None):
    """
    BUILDING REGULARIZATION
    Inputs:
     - satellite image (3 channels)
     - building segmentation (1 channel)
    Output:
     - regularized mask
    """

    img_files = glob.glob(img_folder)
    seg_files = glob.glob(seg_folder)

    img_files.sort()
    seg_files.sort()

    for num, (satellite_image_file, building_segmentation_file) in enumerate(zip(img_files, seg_files)):
        print(satellite_image_file, building_segmentation_file)
        _, rgb_name = os.path.split(satellite_image_file)
        _, seg_name = os.path.split(building_segmentation_file)
        assert rgb_name == seg_name

        output_file = out_folder + seg_name

        E1 = Encoder()
        G = GeneratorResNet()
        G.load_state_dict(torch.load(var.MODEL_GENERATOR))
        E1.load_state_dict(torch.load(var.MODEL_ENCODER))
        E1 = E1.cuda()
        G = G.cuda()

        model = [E1,G]

        M = io.imread(building_segmentation_file)
        M = np.uint16(M)
        P = io.imread(satellite_image_file)
        P = np.uint8(P)

        R = regularization(P, M, model, in_mode=in_mode, out_mode=out_mode)

        if out_mode == "instance":
            io.imsave(output_file, np.uint16(R))
        else:
            io.imsave(output_file, np.uint8(R*255))

        if samples is not None:
            i = 1000
            j = 1000
            h, w = 1080, 1920
            P = P[i:i+h, j:j+w]
            R = R[i:i+h, j:j+w]
            M = M[i:i+h, j:j+w]

            R = mark_boundaries(P, R, mode="thick")
            M = mark_boundaries(P, M, mode="thick")

            R = np.uint8(R*255)
            M = np.uint8(M*255)

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (20,1060)
            fontScale              = 1
            fontColor              = (255,255,0)
            lineType               = 2

            cv2.putText(R, "INRIA dataset, " + rgb_name + ", regularization", 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            cv2.putText(M, "INRIA dataset, " + rgb_name + ", segmentation", 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            io.imsave(samples + "./%d_2reg.png" % num, np.uint8(R))
            io.imsave(samples + "./%d_1seg.png" % num, np.uint8(M))

        copyGeoreference(satellite_image_file, output_file)
        copyGeoreference(satellite_image_file, building_segmentation_file)



regularize_segmentations(img_folder=var.INF_RGB, seg_folder=var.INF_SEG, out_folder=var.INF_OUT, in_mode="semantic", out_mode="instance", samples=None)
