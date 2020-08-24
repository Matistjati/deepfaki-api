import imageio
import numpy as np
from skimage.transform import resize
import sys
import os.path
sys.path.append('dependencies/first-order-model/')
from demo import load_checkpoints
from demo import make_animation
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")


if not os.path.isfile('checkpoints/vox-cpk.pth.tar'):
    raise FileNotFoundError("Could not find training data. Check checkpoints folder")

source_image = imageio.imread("C:/Users/Matis/Desktop/deepfake api/input/obama.jpg")
reader = imageio.get_reader("C:/Users/Matis/Desktop/deepfake api/input/baka mitai driver.mp4")

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
generator, kp_detector = load_checkpoints("dependencies/first-order-model/config/vox-256.yaml",
                                          "C:/Users/Matis/Desktop/deepfake api/checkpoints/vox-cpk.pth.tar",
                                          cpu=False)

predictions = make_animation(source_image, driving_video, generator, kp_detector, cpu=False)

#save resulting video
imageio.mimsave('output/output.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)