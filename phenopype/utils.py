import os
import cv2
import numpy as np
import exifread
from collections import Counter

from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from phenopype.utils_lowlevel import _image_viewer, _construct_yaml_map
from phenopype.settings import colours

#%% settings


#%% modules

class yaml_file_monitor:
    def __init__(self, filepath, **kwargs):
        
        ## kwargs       
        self.flag_print = kwargs.get("print_settings", False)
#        self.flag_update = True
        
        self.dirpath = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)
        self.filepath = filepath
        self.event_handler = PatternMatchingEventHandler(patterns=["*/" + self.filename])
        self.event_handler.on_any_event = self.on_config_update
        
        self.config = load_yaml(self.filepath)
        
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.dirpath, recursive=False)
        self.observer.start()

    def on_config_update(self, event):
        self.config = load_yaml(self.filepath)
        if self.flag_print == True:
            print(self.config, end="")
        self.flag_update = True
        cv2.destroyAllWindows()
        
    def stop(self):
        self.observer.stop()
        self.observer.join()

def load_yaml(filepath):
    SafeConstructor.add_constructor(u'tag:yaml.org,2002:map', _construct_yaml_map)
    yaml = YAML(typ='safe')
    yaml.allow_duplicate_keys = True
    with open(filepath, 'r') as file:
        file = yaml.load(file)
    return file

def exif_date(path): 
    f = open(path, 'rb')
    tags = exifread.process_file(f)
    t = str(tags["EXIF DateTimeOriginal"])
    return t[0:4] + "-" + t[5:7] + "-" + t[8:10] + " " + t[11:20]

def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]

def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

def blur(image, **kwargs):
    blur_kernel = kwargs.get("blur_kernel", 5)
    kern = np.ones((blur_kernel,blur_kernel))/(blur_kernel**2)
    ddepth = -1
    image = cv2.filter2D(image, ddepth,kern)
    return image

def find_skeleton(img):
    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)

    _,thresh = cv2.threshold(img,127,255,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton,iters)
        
        
def find_centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return int(sum_y/length), int(sum_x/length)

def show_img(image, **kwargs):
    """Show one or multiple images by providing path string or array or list of either.
    
    Parameters
    ----------
    image: str, array, or list
        the image or list of images to be displayed. can be path to image (string), array-type, or list of strings or arrays
    window_aspect: str (default: "fixed")
        "fixed" or "free" aspect ratio
    position_reset: bool
        flag whether image positions should be reset when reopening list of images
    position_offset: int 
        if image is list, the distance in pixels betweeen the positions of each newly opened window (only works in conjunction with "position_reset")
    """
    
    ## kwargs
    max_dim = kwargs.get("max_dim", 1980)
    pos_offset = kwargs.get("position_offset", 25)
    pos_reset = kwargs.get("position_reset", False)
    window_aspect = kwargs.get("aspect", "free")
    if window_aspect == "free":
        window_aspect = cv2.WINDOW_NORMAL
    elif window_aspect == "fixed":
        window_aspect = cv2.WINDOW_AUTOSIZE

    
    ## open images list or single images
    while True:
        if isinstance(image, list):
            if len(image)>10:
                warning_banner = np.zeros((30,900,3))
                warning_string = "WARNING: trying to open " + str(len(image)) + " images - proceed (Enter) or stop (Esc)?"
                warning_image = cv2.putText(warning_banner,  warning_string ,(11,22), cv2.FONT_HERSHEY_SIMPLEX  , 0.75, colours.green,1,cv2.LINE_AA)
                cv2.namedWindow('phenopype' ,cv2.WINDOW_AUTOSIZE )
                cv2.imshow('phenopype', warning_image)
                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if k == 27:
                    print("Stop - terminating.")
                    break
                elif k == 13:
                    print("Proceed - Opening images ...")
            idx=0
            for i in image:
                idx+=1
                _image_viewer(i, 
                              mode = "", 
                              window_aspect = window_aspect, 
                              window_name='phenopype' + " - " + str(idx), 
                              window_control="external",
                              max_dim=max_dim)
                if pos_reset == True:
                    cv2.moveWindow('phenopype' + " - " + str(idx),idx+idx*pos_offset,idx+idx*pos_offset)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        else:
            _image_viewer(image, 
                  mode = "", 
                  window_aspect = window_aspect, 
                  window_name='phenopype', 
                  window_control="internal",
                  max_dim=max_dim)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

def get_median_grayscale(image, **kwargs):
    if (image.shape[0] + image.shape[1])/2 > 2000:
        factor = kwargs.get('resize', 0.5)
        image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 
        
    vector = np.ravel(image)
    vector_mc = Counter(vector).most_common(9)
    g = [item[0] for item in vector_mc]
    return int(np.median(g))
    
#        def project_grayscale_finder(self, **kwargs):
#        """Returns median grayscale value from all images inside the project image directory.
#        
#        Parameters
#        -----------
#        
#        resize: in (0.1-1)
#            resize image to increase speed 
#        write: bool, default False
#            write median grayscale to project dataframe
#            
#        """
#        
#        write = kwargs.get('write', False)
#
#        
#        self.gray_scale_list = []
#        for filepath, filename in zip(self.filepaths, self.filenames):
#            image = cv2.imread(filepath,0)
#            med = get_median_grayscale(image)
#            self.gray_scale_list.append(med)
#            print(filename + ": " + str(med))
#            
#        print("\nMean grayscale in directory: " + str(int(np.mean(self.gray_scale_list))))
#        
#        if write == True:
#            self.df["gray_scale"] = self.gray_scale_list



def save_csv(df, name, save_dir, **kwargs):
    """Save a pandas dataframe to csv. 
    
    Parameters
    ----------
    df: df
        object_finder outpur (pandas data frame) to save
    name: str
        name for saved df
    save_dir: str
        location to save df
    append: str (optional)
        append df name with string to prevent overwriting
    overwrite: bool (optional, default: False)
        overwrite df if name exists
    """
    out_dir = save_dir     
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    app = kwargs.get('append',"_results")
    new_name = os.path.splitext(name)[0] + app
        
    df_path=os.path.join(out_dir , new_name  + ".txt")
    
    df = df.fillna(-9999)
    df = df.astype(str)
    
    if kwargs.get('overwrite',True) == False:
        if not os.path.exists(df_path):
            df.to_csv(path_or_buf=df_path, sep=",")

    else:
            df.to_csv(path_or_buf=df_path, sep=",")


def save_img(image, name, **kwargs):
    """Save an image (array) to jpg.
    
    Parameters
    ----------
    image: array
        image to save
    name: str
        name for saved image
    save_dir: str
        location to save image
    append: str ("")
        append image name with string to prevent overwriting
    extension: str ("")
        file extension to save image with
    overwrite: bool (optional, default: False)
        overwrite images if name exists
    """
    # set dir and names
    out_dir = kwargs.get('save_dir', os.getcwd())     
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    app = kwargs.get('append',"")
    new_name = os.path.splitext(name)[0] + app
        
    ext = kwargs.get('extension',os.path.splitext(name)[1])
    new_name = new_name + ext
    
    im_path=os.path.join(out_dir , new_name)
    
    if "resize" in kwargs:
        factor = kwargs.get('resize')
        image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 
    
    if kwargs.get('overwrite',False) == False:
        if not os.path.exists(im_path):
            cv2.imwrite(im_path, image)
    else:
        cv2.imwrite(im_path, image)


