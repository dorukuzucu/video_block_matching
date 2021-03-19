import cv2
import numpy as np
from abc import ABC, abstractmethod


class VideoReader(ABC):
    def __init__(self) -> None:
        self.path = None

    @abstractmethod
    def _read_file(self):
        raise NotImplementedError('Read File Method is not implemented')

    @abstractmethod
    def _read_frames(self):
        raise NotImplementedError('Read File Method is not implemented')

    @abstractmethod
    def __getitem__(self,key):
        raise NotImplementedError('Get Item Method is not implemented')

    @abstractmethod
    def __len__(self):
        pass
    
    
class AviReader(VideoReader):
    def __init__(self,path,mode=None) -> None:
        self.path = path
        self.mode = mode
        self.frames = self._read_frames()
    
    def _read_file(self):
        """
        This method reads avi files to opencv video capture object
        Only .avi extensions are supported
        :return: OpenCV "cap" object
        """
        extension = self.path.split('.')[-1]
        if extension!='avi':
            raise Exception("Invalid Format")

        return cv2.VideoCapture(self.path)

    def _read_frames(self):
        """
        This method reads frames from a vide ocapture object and holds them in memory
        :return: frames as a list
        """
        cap = self._read_file()

        frame_list = []
        ret_list = []

        while True:
            ret, frame = cap.read()
            if ret:
                frame_list.append(np.array(frame))
                ret_list.append(ret)
            else:
                break
        if self.mode=="np":
            frame_list = np.array(frame_list)
        return frame_list

    def __getitem__(self,key):
        return self.frames[key]


    def res(self):
        if self.frames is None:
            raise Exception("Avi file not loaded")

        return self.frames.shape

    def __len__(self):
        if self.frames is None:
            raise Exception("Avi file not loaded")
        return self.frames.shape[0]
    