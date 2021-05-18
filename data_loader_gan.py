import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import random
from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.transform import rotate
import variables as var

TEST = False 

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

class DataLoader():

    def __init__(self, ws=512, nb=10000, bs=8):
        self.nb = nb
        self.bs = bs
        self.ws = ws

        #self.rgb_files = self.rgb_files[:10]
        #self.dsm_files = self.dsm_files[:10]
        #self.gti_files = self.gti_files[:10]

        self.load_data()
        self.num_tiles = len(self.rgb_imgs)
        self.sliding_index = 0

    def generator(self):
        for _ in range(self.nb):
            batch_rgb = []
            batch_gti = []
            batch_seg = []
            for _ in range(self.bs):
                rgb, gti, seg = self.extract_image()

                batch_rgb.append(rgb)

                # the ground truth is categorized
                gti = to_categorical(gti != 0, 2)
                batch_gti.append(gti)

                # the segmentation is categorized
                seg = to_categorical(seg != 0, 2)
                batch_seg.append(seg)

            batch_rgb = np.asarray(batch_rgb)
            batch_gti = np.asarray(batch_gti)
            batch_seg = np.asarray(batch_seg)
            batch_rgb = batch_rgb / 255.0

            #batch_gti = batch_gti[:,:,:,np.newaxis] / 255.0

            yield (batch_rgb, batch_gti, batch_seg)


    def test_shape(self, a):
        ri = a.shape[0] % self.ws
        rj = a.shape[1] % self.ws
        return a[:-ri,:-rj]


    def random_hsv(self, img, value_h=30, value_s=30, value_v=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h = np.int16(h)
        s = np.int16(s)
        v = np.int16(v)

        h += value_h
        h[h < 0] = 0
        h[h > 255] = 255

        s += value_s
        s[s < 0] = 0
        s[s > 255] = 255

        v += value_v
        v[v < 0] = 0
        v[v > 255] = 255

        h = np.uint8(h)
        s = np.uint8(s)
        v = np.uint8(v)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


    def extract_image(self, mode="sequential"):
        if mode is "random":
            rand_t = random.randint(0, self.num_tiles-1)
        else:
            if self.sliding_index < self.num_tiles:
                rand_t = self.sliding_index
                self.sliding_index = self.sliding_index + 1
            else:
                rand_t = 0
                self.sliding_index = 0

        rgb = self.rgb_imgs[rand_t].copy()
        gti = self.gti_imgs[rand_t].copy()
        seg = self.seg_imgs[rand_t].copy()

        h = rgb.shape[1]
        w = rgb.shape[0]

        void = True
        while void:
            rot = random.randint(0,90)
            ri = random.randint(0, int(h-self.ws*2))
            rj = random.randint(0, int(w-self.ws*2))
            win_rgb = rgb[ri:ri+int(self.ws*2), rj:rj+int(self.ws*2)]
            win_gti = gti[ri:ri+int(self.ws*2), rj:rj+int(self.ws*2)]
            win_seg = seg[ri:ri+int(self.ws*2), rj:rj+int(self.ws*2)]
            
            win_rgb = np.uint8(rotate(win_rgb, rot, resize=False, preserve_range=True))
            win_gti = np.uint8(rotate(win_gti, rot, resize=False, preserve_range=True))
            win_seg = np.uint8(rotate(win_seg, rot, resize=False, preserve_range=True))
            
            win_rgb = win_rgb[self.ws//2:-self.ws//2, self.ws//2:-self.ws//2]
            win_gti = win_gti[self.ws//2:-self.ws//2, self.ws//2:-self.ws//2]
            win_seg = win_seg[self.ws//2:-self.ws//2, self.ws//2:-self.ws//2]
            
            if np.count_nonzero(win_seg):
            	void = False

        # Perform some data augmentation
        rot = random.randint(0,3)
        win_rgb = np.rot90(win_rgb, k=rot)
        win_gti = np.rot90(win_gti, k=rot)
        win_seg = np.rot90(win_seg, k=rot)
        if random.randint(0,1) is 1:
            win_rgb = np.fliplr(win_rgb)
            win_gti = np.fliplr(win_gti)
            win_seg = np.fliplr(win_seg)

        r_h = random.randint(-20,20)
        r_s = random.randint(-20,20)
        r_v = random.randint(-20,20)
        win_rgb = self.random_hsv(win_rgb, r_h, r_s, r_v)

        win_rgb = win_rgb.astype(np.float32)
        win_gti = win_gti.astype(np.float32)
        win_seg = win_seg.astype(np.float32)
        return (win_rgb, win_gti, win_seg)

        
    def load_data(self):
        self.rgb_imgs = []
        self.gti_imgs = []
        self.seg_imgs = []

        rgb_files = glob(var.DATASET_RGB)
        gti_files = glob(var.DATASET_GTI)
        seg_files = glob(var.DATASET_SEG)

        rgb_files.sort()
        gti_files.sort()
        seg_files.sort()

        combined = list(zip(rgb_files, gti_files, seg_files))
        random.shuffle(combined)

        rgb_files[:], gti_files[:], seg_files[:] = zip(*combined)

        if TEST:
            rgb_files = rgb_files[:4]
            gti_files = gti_files[:4]
            seg_files = seg_files[:4]

        for rgb_name, gti_name, seg_name in tqdm(zip(rgb_files, gti_files, seg_files), total=len(rgb_files), desc="Loading dataset into RAM"):

            tmp = io.imread(rgb_name)
            tmp = tmp.astype(np.uint8)
            self.rgb_imgs.append(tmp)

            tmp = io.imread(gti_name)
            tmp = tmp.astype(np.uint8)
            self.gti_imgs.append(tmp)

            tmp = io.imread(seg_name)
            tmp = tmp.astype(np.uint8)
            self.seg_imgs.append(tmp)

