from PIL import Image
from io import BytesIO
import tensorboardX as tb
from tensorboardX import SummaryWriter
from tensorboardX.summary import Summary
import numpy as np

class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = SummaryWriter(model_dir)
    def add_image(self, tag, img, step):
        ''' Expects channels last rgb image '''
        img = np.array(img)
        if len(img.shape) == 2:
            img = Image.fromarray(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        self.summary_writer.add_image(tag, img)

    def add_scalar(self, tag, value, step):
        self.summary_writer.add_scalar(tag, value, step)

    def add_text(self, tag, text, step):
        self.summary_writer.add_text(tag, text, step)
