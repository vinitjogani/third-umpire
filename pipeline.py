from detection.neuralnet.cnn_dcsn import pipeline as cnn
from interpolation.video_to_slomo import main as slow
import utils

def pipeline(vid):
    slow(vid, 8, 'slow.mp4')
    utils.writer(cnn('slow.mp4'), 'out.mp4')

pipeline('dataset/Guptill_Trim.mp4')