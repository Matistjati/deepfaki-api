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
import pickle
import moviepy.editor as mp


class DeepFakeApi:
    @staticmethod
    def __init__():
        DeepFakeApi.initialized = True

        if not os.path.isfile('checkpoints/vox-cpk.pth.tar'):
            raise FileNotFoundError("Could not find training data. Check checkpoints folder")

        print("Loading checkpoints")
        DeepFakeApi.generator, DeepFakeApi.kp_detector = load_checkpoints("dependencies/first-order-model/config/vox-256.yaml",
                                                                          "checkpoints/vox-cpk.pth.tar",
                                                                          cpu=False)

    @staticmethod
    def generate_deepfake(image_path, driver_path, sound_path, output_path="output/output.mp4"):
        if not hasattr(DeepFakeApi, "initialized"):
            DeepFakeApi.__init__()

        print("Loading driver and source image")
        source_image = imageio.imread(image_path)
        reader = imageio.get_reader(driver_path)
        meta_data = reader.get_meta_data()

        fps = reader.get_meta_data()['fps']

        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        print("Resizing driver and source image")
        source_image = resize(source_image, (256, 256))[..., :3]

        # Resize only if necessary
        if meta_data['source_size'] != (256, 256) and meta_data['size'] != (256, 256):
            driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        else:
            # We still need to remap the color values from [0,256] to [0,1]
            driving_video = [(frame * (1 / 256))[..., :3] for frame in driving_video]


        print("Creating animation")
        predictions = make_animation(source_image, driving_video, DeepFakeApi.generator, DeepFakeApi.kp_detector, cpu=False)

        # save resulting video
        print("Saving output")
        clips = [mp.ImageClip(img_as_ubyte(m)).set_duration(1 / fps)
                 for m in predictions]

        concat_clip = mp.concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(output_path, audio=sound_path, fps=fps)


if __name__ == "__main__":
    DeepFakeApi.generate_deepfake("input/obama.jpg", "drivers/baka mitai driver.mp4", "drivers/baka mitai.mp3", "output/output.mp4")


