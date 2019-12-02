#%%
import copy
import cv2
import os
import platform
import subprocess

from phenopype import segmentation
from phenopype.utils import yaml_file_monitor
from phenopype.utils_lowlevel import _image_viewer

#%%

class pype:
    def __init__(self, image):
        ## load image
        if isinstance(image, str):
            image = cv2.imread(image)  
        if len(image.shape)==2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.image = image
        
    def run(self, filepath, **kwargs):
        
        ## kwargs
        print_settings = kwargs.get("print_settings",False)   
        
        ## open config file with system viewer
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', filepath))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(filepath)
        else:                                   # linux variants
            subprocess.call(('xdg-open', filepath))
        
        self.fm = yaml_file_monitor(filepath, print_settings=print_settings)
        self.prev_view = {}
        
        while True:       
            self.image_mod = copy.deepcopy(self.image)
            config = self.fm.config
            
            for step in ["preprocessing", "segmentation", "organization"]:
                if step:
                    pass
                config_step = config.get(step)
                try:
                    if isinstance(config_step, list):
                        for method_item in config_step:
                            method_name = step + "." + method_item[0]
                            method_args = method_item[1]
                            method = eval(method_name) 
                            if method_args:
                                self.image_mod = method(self.image_mod, **method_args) 
                            else:
                                self.image_mod = method(self.image_mod) 
                    elif isinstance(config_step, dict):
                        for method_name, method_args in config_step.items():
                            method_name = step + "."+  method_name
                            method = eval(method_name) 
                            if method_args:
                                self.image_mod = method(self.image_mod, **method_args) 
                            else:
                                self.image_mod = method(self.image_mod) 
                except Exception as e: print(e)
            
            iv = _image_viewer(self.image_mod, prev_attributes=self.prev_view, max_dim=1000)
            self.prev_view = iv.__dict__
            continue     
            
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                self.fm.observer.stop()
                self.fm.observer.join()
                break
            elif cv2.waitKey(0) == 13:
                cv2.destroyAllWindows()
                self.fm.observer.stop()
                self.fm.observer.join()
                break
                