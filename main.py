from mask_detect import Mask_detect
import os, sys
import cv2
from threading import Thread
from yolo.models.experimental import attempt_load
from yolo.utils.torch_utils import select_device
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "yolo"))

global Camfeed, feed
Camfeed = {}
feed = {}
global mask
mask_det = Mask_detect()

class Cam:
    def __init__(self,camera, camname):
        self.cam = camera
        self.camname = camname

    def CameraStream(self):

        try:
            os.mkdir('feed')
        except Exception as e:
            print('feed dir is already there')

        vid = cv2.VideoCapture(self.cam)
        while(True):

            ret, frame = vid.read()
            if ret == True:
                global feed
                global Camfeed
                Camfeed[self.camname] = frame.copy()
                feed[self.camname] = True
                # print(feed)
                # cv2.imshow('frame', frame)
                # print('Feed is coming')
                cv2.imwrite('feed/'+str(self.camname)+'.jpg', frame)
            else:
                feed[self.cam] = False
                # print('No feed')

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        vid.release()
        cv2.destroyAllWindows()

class Mask:

    def __init__(self):
        self.device = select_device('0')
        self.model_mask = None
        self.loaded = False

    def load_model(self,weights_path):
        try:
            self.model_mask = attempt_load(weights_path,map_location=self.device)
            return True
        except Exception as e:
            print('model load exception>>:', e)
            pass

    def run_model(self, weights_path, cameras):

        self.cams = cameras

        self.loaded = self.load_model(weights_path)

        if self.loaded:
            try:
                os.mkdir('results')
            except Exception as e:
                print('results dir is already there')
            for cam in self.cams:
                try:
                    os.mkdir('results/'+str(cam))
                except Exception as e:
                    print('results/'+str(cam)+'dir is already there')

            while True:

                for cam in self.cams:
                    # print('>>>>>>>>>>',feed)
                    if feed[cam]:
                        try:
                            print('running model for ', cam)
                            global Camfeed
                            mask_det.run(Camfeed[cam].copy(), self.model_mask, cam)
                        except Exception as e:
                            print('mask model run exception>>:', e)
                            # raise e


if __name__ == "__main__":
    f = open('config.json')

    data = json.load(f)
    cameras = data['cameras']
    mask_weights = data['weights']

    Mask1 = Mask()
    cam_objs = []

    feed = {x: False for x in cameras.keys()}

    for x in cameras.keys():
        cam_objs.append(Cam(cameras[x], x))

    for cam in cam_objs:
        Thread(target = cam.CameraStream).start()

    Mask1.run_model(mask_weights, cameras.keys())
