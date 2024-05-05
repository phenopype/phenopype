#%% modules

import copy
import numpy as np
import numpy.ma as ma
import pandas as pd
import cv2
import os
import pprint
import sys

from math import inf

from phenopype import _vars
from phenopype import utils_lowlevel
from phenopype.core import preprocessing
from phenopype.core import segmentation
from phenopype.core import visualization

#%% classes


class motion_tracker(object):
    """
    Initialize motion tracker class; extract information (length, fps, codec, 
    etc.) from input video and pass to other tracking methods.

    Parameters
    ----------
    video_path : str
        path to video
    at_frame : int, optional
        frame index to be used to extract the video information
        
    """

    def __init__(self, video_path, at_frame=1):

        ## extract frame
        if os.path.isfile(video_path):
            capture = cv2.VideoCapture(video_path)
            idx = 0
            while capture.isOpened():
                idx += 1
                if idx == at_frame:
                    ret, frame = capture.read()
                    break
                else:
                    capture.grab()
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        else:
            print("No compatible video file found under provided path")
            return

        ## properties
        self.path = video_path
        self.name = os.path.basename(self.path)
        self.nframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.fourcc_str = utils_lowlevel._decode_fourcc(
            capture.get(cv2.CAP_PROP_FOURCC)
        )
        self.length = str(
            str(int((self.nframes / self.fps) / 60)).zfill(2)
            + ":"
            + str(
                int(
                    (
                        ((self.nframes / self.fps) / 60)
                        - int((self.nframes / self.fps) / 60)
                    )
                    * 60
                )
            ).zfill(2)
        )
        self.dimensions = tuple(reversed(frame.shape[0:2]))
        if frame.shape[2] == 3:
            self.is_colour = True

        ## for masks
        self.image = frame
        self.df_image_data = pd.DataFrame(
            {
                "filename": self.name,
                "filepath": self.path,
                "filetype": self.fourcc_str,
                "width": self.dimensions[0],
                "height": self.dimensions[1],
            },
            index=[0],
        )

        ## default
        self.flag_save_video = False

        ## release capture
        capture.release()

        ## feeddback
        print("\n")
        print("--------------------------------------------------------------")
        print('Input video properties - "' + self.name + '":\n')
        print(
            "Frames per second: "
            + str(self.fps)
            + "\nN frames: "
            + str(self.nframes)
            + "\nLength: "
            + str(self.length)
            + " (mm:ss)"
            + "\nDimensions: "
            + str(self.dimensions)
            + "\nColour video: "
            + str(self.is_colour)
            + "\nFourCC code: "
            + str(self.fourcc_str)
        )
        print("--------------------------------------------------------------")

    def video_output(
        self,
        video_format=None,
        save_suffix="out",
        dirpath=None,
        fps=None,
        save_colour=None,
        dimensions=None,
        resize=1,
    ):
        """
        Set properties of output video file. Most settings can be left blank, 
        so that settings from the input video will be applied
        
        Parameters
        ----------
        video_format: str, optional
            format of the output video file. needs to be a fourcc-string
            https://www.fourcc.org/codecs.php
        save_suffix: str, optional
            name for the output video file (defaut: "input-filename" + _out.avi)
        dirpath: str, optional
            save directory for the output video file. will be created if not existing
        fps: int, optional
            frames per second of the output video file
        dimensions: tuple, optional
            dimensions (width, length) of the output video file
        resize: int, optional
            factor by which to resize the output video dimensions
        save_colour: bool, optional
            should the saved video frames be in colour 
        """

        self.flag_save_video = True

        ## name and path
        name = (
            os.path.splitext(self.name)[0]
            + "_"
            + save_suffix
            + os.path.splitext(self.name)[1]
        )
        if dirpath.__class__.__name__ == "NoneType":
            dirpath = os.path.dirname(self.path)
        else:
            if not os.path.isdir(dirpath):
                q = input("Save folder {} does not exist - create?.".format(dirpath))
                if q in ["True", "true", "y", "yes"]:
                    os.makedirs(dirpath)
                else:
                    print("Directory not created - aborting")
                    return
        path_out = os.path.join(dirpath, name)

        ## video properties
        if video_format.__class__.__name__ == "NoneType":
            fourcc_str = self.fourcc_str
        else:
            fourcc_str = video_format
        fourcc_out = cv2.VideoWriter_fourcc(*fourcc_str)
        if fps.__class__.__name__ == "NoneType":
            fps_out = self.fps
        if dimensions.__class__.__name__ == "NoneType":
            dimensions_out = self.dimensions
        if save_colour.__class__.__name__ == "NoneType":
            self.flag_save_colour = self.is_colour
        else:
            self.flag_save_colour = save_colour

        self.resize_factor = resize
        if self.resize_factor < 1:
            res_msg = " (original resized by factor " + str(self.resize_factor) + ")"
        else:
            res_msg = ""
        dimensions_out = (
            int(dimensions_out[0] * self.resize_factor),
            int(dimensions_out[1] * self.resize_factor),
        )

        ## start video-writer
        self.writer = cv2.VideoWriter(
            path_out, fourcc_out, fps_out, dimensions_out, self.flag_save_colour
        )

        print("\n")
        print("--------------------------------------------------------------")
        print('Output video settings - "' + self.name + '":\n')
        print(
            "Save name: "
            + name
            + "\nSave dir: "
            + os.path.abspath(dirpath)
            + "\nFrames per second: "
            + str(fps_out)
            + "\nDimensions: "
            + str(dimensions_out)
            + res_msg
            + "\nColour video: "
            + str(self.flag_save_colour)
            + "\nFormat (FourCC code): "
            + fourcc_str
        )
        print("--------------------------------------------------------------")

    def detection_settings(
        self,
        skip=5,
        warmup=0,
        start_after=0,
        finish_after=0,
        history=60,
        threshold=10,
        detect_shadows=True,
        mode="MOG",
        methods=None,
        masks=None,
        c_mask=False,
        c_mask_shape="rect",
        c_mask_size=50,
    ):

        """
        Set properties of output video file. Most settings can be left at their 
        default value.
        
        Parameters
        ----------
        skip: int, optional
            how many frames to skip between each capture
        warmup: int, optional
            warmup period in seconds for the background subtractor
        start_after: int, optional
            start after X seconds
        finish_after: int, optional
            finish after X seconds
        history: int, optional
            how many frames to use for fg-bg subtraction algorithm
        threshold: int, optional
            sensitivity-level for fg-bg subtraction algorithm (lower = more 
            sensitive)
        detect_shadows: bool, optional
            attempt to detect shadows - will be returned as gray pixels
        mode: {"MOG", "KNN"} str, optional
            type of fg-bg subtraction algorithm
        methods: method or list of methods, optional
            list with tracking_method objects
        c_mask: bool, optional 
            consecutive masking. if multiple methods are defined, the objects 
            detected first will mask the objects detected in subsequent methods
        c_mask_shape: {"rect", "ellipse", "contour"} str, optional
            which shape should the consecutive mask have
        c_mask_size: int, optional
            area in pixels that is added around the mask
            
        """
        ## kwargs
        self.skip = skip
        self.warmup = warmup
        self.start = start_after
        self.finish = finish_after
        self.flag_detect_shadows = detect_shadows
        self.flag_consecutive = c_mask
        self.consecutive_shape = c_mask_shape
        self.consecutive_size = c_mask_size
        self.masks = masks

        ## select background subtractor
        if mode == "MOG":
            self.fgbg_subtractor = cv2.createBackgroundSubtractorMOG2(
                int(history * (self.fps / self.skip)),
                threshold,
                self.flag_detect_shadows,
            )
        elif mode == "KNN":
            self.fgbg_subtractor = cv2.createBackgroundSubtractorKNN(
                int(history * (self.fps / self.skip)),
                threshold,
                self.flag_detect_shadows,
            )

        ## return settings of methods
        if not methods.__class__.__name__ == "NoneType":
            if methods.__class__.__name__ == "tracking_method":
                methods = [methods]

            self.methods = methods
            for m in self.methods:
                print("-----------------------------")
                print("Detection method \"{}\":\n".format(m.label))
                m._print_settings()
                print("-----------------------------")
                print("\n")

        print("\n")
        print("--------------------------------------------------------------")
        print('Motion detection settings - "' + self.name + '":\n')
        print(
            "Background-subtractor: "
            + str(mode)
            + "\nHistory: "
            + str(history)
            + " seconds"
            + "\nSensitivity: "
            + str(threshold)
            + "\nRead every nth frame: "
            + str(self.skip)
            + "\nDetect shadows: "
            + str(self.flag_detect_shadows)
            + "\nStart after n seconds: "
            + str(self.start)
            + "\nFinish after n seconds: "
            + str(self.finish if self.finish > 0 else " - ")
        )
        print("--------------------------------------------------------------")

    def run_tracking(self, feedback=True, canvas="overlay", overlay_weight=0.5, **kwargs):
        """
        Start motion tracking procedure. Enable or disable video feedback, 
        output and select canvas (overlay of detected objects, foreground mask,
        input video).
        
        Parameters
        ----------
        feedback: bool, optional
            show output of tracking
        canvas: {"overlay","fgmask","input"} str, optional
            the background for the ouput video
        overlay_weight: float (default: 0.5)
            if canvas="overlay", how transparent should the overlay should be

        """

        ## kwargs
        flag_feedback = feedback

        ## check if settings have been called
        if not hasattr(self, "fgbg_subtractor"):
            self.detection_settings()

        ## initialize
        self.df = pd.DataFrame()
        self.idx1, self.idx2 = (0, 0)
        self.capture = cv2.VideoCapture(self.path)
        self.start_frame = int(self.start * self.fps)
        if self.finish > 0:
            self.finish_frame = int(self.finish * self.fps)
        else:
            self.finish_frame = self.nframes

        if all(hasattr(self, attr) for attr in ["methods", "masks"]):
            for m in self.methods:
                m._apply_masks(frame=self.image, masks=self.masks)

        ## loop thrpugh frames
        while self.capture.isOpened():

            ## frame indexing
            self.idx1, self.idx2 = (self.idx1 + 1, self.idx2 + 1)
            if self.idx2 == self.skip:
                self.idx2 = 0

            ## time conversion
            mins = str(int((self.idx1 / self.fps) / 60)).zfill(2)
            secs = str(int((((self.idx1 / self.fps) / 60) - int(mins)) * 60)).zfill(2)
            self.time_stamp = (
                "Time: "
                + mins
                + ":"
                + secs
                + "/"
                + self.length
                + " - Frames: "
                + str(self.idx1)
                + "/"
                + str(int(self.nframes))
            )

            ## end-of-frames-control
            if self.idx1 == self.finish_frame - 1:
                self.capture.release()
                if self.flag_save_video:
                    self.writer.release()
                break

            ## capture frame
            if (
                self.idx1 > self.start_frame - int(self.warmup * self.fps)
                and self.idx2 == 0
            ):
                self.ret, self.frame = self.capture.read()

                ## skip empty frames
                if self.ret == False:
                    continue
                else:
                    capture_frame = True
                    if (
                        self.idx1 < self.start_frame
                        and self.idx1 > self.start_frame - int(self.warmup * self.fps)
                    ):
                        print(self.time_stamp + " - warmup")
                    else:
                        print(self.time_stamp + " - captured")
            else:
                self.capture.grab()
                print(self.time_stamp)
                continue

            ## if captured, apply masks > apply methods > write to output
            if capture_frame == True:

                # initiate tracking
                fgmask = self.fgbg_subtractor.apply(self.frame)
                fgmask_copy = copy.deepcopy(fgmask)
                self.frame_overlay = self.frame

                # apply methods
                if "methods" in vars(self):
                    idx = 0
                    for m in self.methods:
                        fgmask_copy = copy.deepcopy(fgmask)
                        (
                            self.fgmask_mod,
                            self.overlay,
                            self.method_contours,
                            self.frame_df,
                        ) = m._run(frame=self.frame, fgmask=fgmask_copy)
                        idx += 1

                        # shadowing of methods
                        if self.flag_consecutive and idx < len(self.methods):
                            self.method_mask = np.zeros_like(fgmask_copy)
                            for contour in self.method_contours:
                                if self.consecutive_shape == "contour":
                                    self.method_mask = cv2.drawContours(
                                        self.method_mask,
                                        [contour],
                                        0,
                                        utils_lowlevel._get_bgr("white"),
                                        -1,
                                    )  # Draw filled contour in mask
                                elif self.consecutive_shape == "ellipse":
                                    self.method_mask = cv2.ellipse(
                                        self.method_mask,
                                        cv2.fitEllipse(contour),
                                        utils_lowlevel._get_bgr("white"),
                                        -1,
                                    )
                                elif self.consecutive_shape in ["rect", "rectangle"]:
                                    rx, ry, rw, rh = cv2.boundingRect(contour)
                                    cv2.rectangle(
                                        self.method_mask,
                                        (int(rx), int(ry)),
                                        (int(rx + rw), int(ry + rh)),
                                        utils_lowlevel._get_bgr("white"),
                                        -1,
                                    )
                                kernel = cv2.getStructuringElement(
                                    cv2.MORPH_RECT,
                                    (self.consecutive_size, self.consecutive_size),
                                )
                                self.method_mask = cv2.dilate(
                                    self.method_mask, kernel, iterations=1
                                )
                            fgmask = cv2.subtract(fgmask, self.method_mask)

                        # create overlay for each method
                        self.frame_overlay = cv2.addWeighted(
                            self.frame_overlay, 1, self.overlay, overlay_weight, 0
                        )

                        # make data.frame
                        self.frame_df.insert(0, "frame_abs", self.idx1)
                        self.frame_df.insert(
                            1, "frame", int((self.idx1 - self.start_frame) / self.skip)
                        )
                        self.frame_df.insert(2, "mins", mins)
                        self.frame_df.insert(3, "secs", secs)
                        # self.df = self.df.append(
                        #     self.frame_df, ignore_index=True, sort=False
                        # )
                        self.df = pd.concat([self.df, self.frame_df], ignore_index=True, sort=False)


                ## select canvas
                if "methods" in vars(self):
                    if canvas == "overlay":
                        self.canvas = self.frame_overlay
                    elif canvas == "fgmask":
                        self.canvas = fgmask_copy
                    elif canvas == "fgmask_mod":
                        self.canvas = self.fgmask_mod
                    else:
                        self.canvas = self.frame
                else:
                    self.canvas = fgmask_copy

                ## resize
                if "resize_factor" in vars(self):
                    self.canvas = cv2.resize(
                        self.canvas,
                        (0, 0),
                        fx=self.resize_factor,
                        fy=self.resize_factor,
                    )

                ## convert to colour
                if len(self.canvas.shape) < 3:
                    self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)

                ## draw masks
                if not self.masks.__class__.__name__ == "NoneType":
                    for key, value in self.masks[_vars._mask_type].items():
                        self.canvas = visualization.draw_mask(
                            self.canvas, {_vars._mask_type: {key: value}}, label=True
                        )

                ## feedback
                if flag_feedback == True:
                    cv2.namedWindow("phenopype", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("phenopype", self.canvas)

                # save output
                if self.flag_save_video == True:
                    if self.flag_save_colour and len(self.canvas.shape) < 3:
                        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
                    self.writer.write(self.canvas)

            ## keep stream open
            if cv2.waitKey(1) & 0xFF == 27:
                self.capture.release()
                self.writer.release()
                break

        ## cleanup
        self.capture.release()
        if "writer" in vars(self):
            self.writer.release()
        cv2.destroyAllWindows()

        ## return DataFrame
        debug =  kwargs.get("debug", False)
        
        if debug:
            return self
        else:
            return self.df


class tracking_method:

    """
    Constructs a tracking method that can be supplied to the motion_tracker 
    class.
    
    Parameters
    ----------
    label: str, optional
        label for all objects detected by this method
    blur: int, optional
        blurring of fgbg-mask (kernel size)
    threshold: int, optional
        binarization of fgbg-mask after blurring (threshold value)
    remove_shadows: bool, optional
        if motion_tracker has detect_shadows=True, they can be removed here
    min_area : int, optional
        minimum contour area in pixels to be included
    max_area : int, optional
        maximum contour area in pixels to be included
    min_length : int, optional
        minimum diameter of boundary circle to be included
    max_length : int, optional
        maximum diameter of boundary circle to be included
    mode: {"single", "multiple"} str, optional
        track "multiple", or "single" (biggest by diameter) objects
    remove_shadows: bool, optional
        remove shadows if shadow-detection is actived in MOG-algorithm
    overlay_colour: {"red", "green", "blue", "black", "white"} str, optional
        which colour should tracked objects have
    operations: list (default: ["length", "area"])
        determines the type of operations to be performed on the detected objects:
            - "diameter" of the bounding circle of our object
            - "area" within the contour of our object
            - "grayscale" mean and standard deviation of grayscale pixel values 
              inside the object contours
            - "grayscale_background" background within boundingbox of contour
            - "bgr" mean and standard deviation of blue, green and red pixel 
              values inside the object contours
    """

    def __init__(
        self,
        label="m1",
        blur=5,
        threshold=127,
        remove_shadows=True,
        min_length=0,
        max_length=inf,
        min_area=0,
        max_area=inf,
        mode="multiple",
        overlay_colour="red",
        operations=[],
    ):

        ## kwargs
        self.blur_kernel = blur
        self.label = label
        self.overlay_colour = utils_lowlevel._get_bgr(overlay_colour)
        self.min_length, self.max_length = min_length, max_length
        self.min_area, self.max_area = min_area, max_area
        self.mode = mode
        self.operations = operations
        self.threshold_value = threshold
        self.remove_shadows = remove_shadows

    def _apply_masks(self, frame, masks):
        """
        Applies masks drawn using the motion_tracker.
        
        Internal reference - don't call this directly. 
        """
        if masks.__class__.__name__ == "dict":
            if _vars._mask_type in masks:
                self.mask_bool = {}
                for key, value in masks[_vars._mask_type].items():
                    polygons = masks[_vars._mask_type][key]["data"][
                        _vars._mask_type
                    ]
                    label = masks[_vars._mask_type][key]["data"]["label"]
                    for coords in polygons:
                        mask_bool = utils_lowlevel._create_mask_bool(frame, coords)
                        self.mask_bool[label] = mask_bool

    def _print_settings(self, width=30, indent=1, compact=True):
        """
        Prints the settings of the tracking method. 
        
        Internal reference - don't call this directly. 
        """

        print_dict = {}
        for key, value in self.__dict__.items():
            if type(value) in [str, float, int, tuple, bool]:
                print_dict[key] = value
                
        pretty = pprint.PrettyPrinter(width=width, compact=compact, indent=indent)
        pretty.pprint(print_dict)

    def _run(self, frame, fgmask):
        """
        Run tracking method on current frame. 
        
        Internal reference - don't call this directly.      
        """

        ## initialize
        self.overlay = np.zeros_like(frame)
        self.overlay_bin = np.zeros(frame.shape[0:2], dtype=np.uint8)
        self.frame_df = pd.DataFrame()

        if self.remove_shadows == True:
            ret, fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

        ## blur
        fgmask = preprocessing.blur(fgmask, self.blur_kernel)

        # ## threshold
        fgmask = segmentation.threshold(
            fgmask, method="binary", invert=True, value=self.threshold_value
        )

        ## find contours
        contours, hierarchy = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ## perform operations on contours
        if len(contours) > 0:
            list_contours, list_area, list_length, list_center_coordinates = (
                [],
                [],
                [],
                [],
            )
            df_list, df_column_names = [], []

            # check if contour matches min/max length provided
            for contour in contours:
                if contour.shape[0] > 4:
                    center, radius = cv2.minEnclosingCircle(contour)
                    length = int(radius * 2)
                    area = int(cv2.contourArea(contour))
                    if all(
                        [
                            length > self.min_length and length < self.max_length,
                            area > self.min_area and area < self.max_area,
                        ]
                    ):
                        list_length.append(length)
                        list_area.append(area)
                        list_contours.append(contour)
                        list_center_coordinates.append(center)

            if len(list_contours) > 0:
                # if single biggest contour:
                if self.mode == "single":
                    if len(contours) == 1:
                        pass
                    elif len(contours) > 1:
                        max_idx = np.argmax(list_length)
                        list_contours = [list_contours[max_idx]]
                        list_length = [list_length[max_idx]]
                        list_area = [list_area[max_idx]]
                        list_center_coordinates = [list_center_coordinates[max_idx]]

                list_x, list_y = [], []
                list_grayscale, list_grayscale_background = [], []
                list_b, list_g, list_r = [], [], []
                list_mask_check = []

                for contour, center in zip(list_contours, list_center_coordinates):

                    # operations
                    x = int(center[0])
                    y = int(center[1])
                    list_x.append(x)
                    list_y.append(y)

                    if "mask_bool" in vars(self):
                        temp_list = []
                        for key, val in self.mask_bool.items():
                            temp_list.append(val[y, x])
                        list_mask_check.append(temp_list)

                    rx, ry, rw, rh = cv2.boundingRect(contour)
                    frame_roi = frame[ry : ry + rh, rx : rx + rw]
                    frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                    mask_roi = fgmask[ry : ry + rh, rx : rx + rw]

                    if any("grayscale" in o for o in self.operations):
                        grayscale = ma.array(
                            data=frame_roi_gray, mask=np.logical_not(mask_roi)
                        )
                        list_grayscale.append(int(np.mean(grayscale)))

                    if any("grayscale_background" in o for o in self.operations):
                        grayscale_background = ma.array(
                            data=frame_roi_gray, mask=mask_roi
                        )
                        if not grayscale_background.mask.all():
                            list_grayscale_background.append(
                                int(np.mean(grayscale_background))
                            )
                        else:
                            list_grayscale_background.append(9999)

                    if any("bgr" in o for o in self.operations):
                        b = ma.array(
                            data=frame_roi[:, :, 0], mask=np.logical_not(mask_roi)
                        )
                        list_b.append(int(np.mean(b)))
                        g = ma.array(
                            data=frame_roi[:, :, 1], mask=np.logical_not(mask_roi)
                        )
                        list_g.append(int(np.mean(g)))
                        r = ma.array(
                            data=frame_roi[:, :, 2], mask=np.logical_not(mask_roi)
                        )
                        list_r.append(int(np.mean(r)))

                    # drawing
                    self.overlay = cv2.drawContours(
                        self.overlay, [contour], 0, self.overlay_colour, -1
                    )  # Draw filled contour in mask
                    self.overlay = cv2.putText(
                        self.overlay,
                        self.label,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        self.overlay_colour,
                        1,
                        cv2.LINE_AA,
                    )
                    self.overlay = cv2.rectangle(
                        self.overlay,
                        (rx, ry),
                        (int(rx + rw), int(ry + rh)),
                        self.overlay_colour,
                        2,
                    )

                df_list = df_list + [list_x]
                df_list = df_list + [list_y]
                df_column_names = df_column_names + ["x", "y"]

                if any("diameter" in o for o in self.operations):
                    df_list = df_list + [list_length]
                    df_column_names.append("diameter")

                if any("area" in o for o in self.operations):
                    df_list = df_list + [list_area]
                    df_column_names.append("area")

                if any("grayscale" in o for o in self.operations):
                    df_list = df_list + [list_grayscale]
                    df_column_names.append("grayscale")

                if any("grayscale_background" in o for o in self.operations):
                    df_list = df_list + [list_grayscale_background]
                    df_column_names.append("grayscale_background")

                if any("bgr" in o for o in self.operations):
                    df_list = df_list + [list_b]
                    df_list = df_list + [list_g]
                    df_list = df_list + [list_r]
                    df_column_names = df_column_names + ["b", "g", "r"]

                frame_df = pd.DataFrame(data=df_list)
                frame_df = frame_df.transpose()
                frame_df.columns = df_column_names
                frame_df["label"] = self.label

                if "mask_bool" in vars(self):

                    mask_df = pd.DataFrame(list_mask_check, columns=[*self.mask_bool])
                    self.frame_df = pd.concat(
                        [frame_df.reset_index(drop=True), mask_df], axis=1
                    )

                else:
                    self.frame_df = frame_df

                self.contours = list_contours

                return fgmask, self.overlay, self.contours, self.frame_df

            else:
                frame_df = pd.DataFrame()
                return fgmask, self.overlay, [], self.frame_df

        else:
            frame_df = pd.DataFrame()
            return fgmask, self.overlay, [], self.frame_df
