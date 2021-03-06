#%% modules
import cv2, copy
import numpy as np
import pandas as pd

from math import sqrt
import numpy.ma as ma

from phenopype.settings import colours
from phenopype.utils import load_image_data
from phenopype.utils_lowlevel import _image_viewer, _contours_arr_tup, _equalize_histogram
from phenopype.utils_lowlevel import _auto_text_size, _auto_text_width

#%% functions


def create_mask(
    obj_input,
    df_masks=None,
    df_image_data=None,
    include=True,
    overwrite=False,
    edit=False,
    canvas="image",
    label="mask1",
    tool="rectangle",
    max_dim=None,
    **kwargs
):
    """
    Draw rectangle or polygon mask onto image by clicking and dragging the 
    cursor over the image. One mask can contain multiple sets of coordinates, 
    i.e. multiple and not overallping masks. For rectangle mask, finish with 
    ENTER. For polygons, finish current polygon with CTRL, and then with ENTER.
    
    Parameters
    ----------
    obj_input : array or container
        input object
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata, will be added to mask
        output DataFrame
    include: bool, optional
        determine whether resulting mask is to include or exclude objects within
    label: str, optinal
        assigns a label to the mask
    overwrite: bool, optional
        if working using a container, or from a phenopype project directory, 
        should existing masks with the same label be overwritten
    tool: {"rectangle", "polygon"} str, optional
        select tool by which mask is drawn

    Returns
    -------
    df_masks: DataFrame or container
        contains mask coordiantes
    """

    ## kwargs
    flag_overwrite = overwrite
    flag_edit = edit
    flag_canvas = canvas
    test_params = kwargs.get("test_params", None)

    prev_masks = {}

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame(
                {"filename": "unknown"}, index=[0]
            )  
    elif obj_input.__class__.__name__ in ["container", "motion_tracker"]:
        if flag_canvas == "image":
            image = copy.deepcopy(obj_input.image)
        elif flag_canvas == "canvas":
            image = copy.deepcopy(obj_input.canvas)
        if hasattr(obj_input, "df_image_data"):
            df_image_data = copy.deepcopy(obj_input.df_image_data)
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)
            if "index" in df_masks:
                df_masks.drop(columns = ["index"], inplace=True)
            df_masks.reset_index(drop=True, inplace=True)
    else:
        print("wrong input format.")
        return

    ## check if exists
    while True:
        if not df_masks.__class__.__name__ == "NoneType":
            df_masks_sub = df_masks.loc[df_masks["mask"] == label]
            df_masks_sub = df_masks_sub[
                df_masks_sub.columns.intersection(["mask", "include","coords"])
            ]  
        if not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == False and flag_edit == False:
            if label in df_masks_sub["mask"].values:
                print("- mask with label " + label + " already created (edit/overwrite=False)")
                break
        elif not df_masks.__class__.__name__ == "NoneType" and flag_edit == True:
            if label in df_masks_sub["mask"].values:
                prev_point_list = []
                prev_rect_list = []
                for index, row in df_masks_sub.iterrows():
                    coords = eval(row["coords"])
                    prev_point_list.append(coords)
                    if tool == "rect" or tool == "rectangle":
                        prev_rect_list.append([coords[0][0], coords[0][1], coords[2][0], coords[2][1]])
                prev_masks = {"point_list": prev_point_list,
                              "rect_list": prev_rect_list}
                df_masks = df_masks.drop(df_masks[df_masks["mask"] == label].index)
                print("- creating mask (editing)")
        elif not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == True:
            if label in df_masks["mask"].values:
                ## remove rows from original drawing df
                df_masks = df_masks.drop(df_masks[df_masks["mask"] == label].index)
                print("- creating mask (overwriting)")
                pass
        elif df_masks.__class__.__name__ == "NoneType":
            df_masks = pd.DataFrame(columns=["mask", "include", "coords"])
            print("- creating mask")
            pass

        ## method
        if not test_params.__class__.__name__ == "NoneType":
            out = _image_viewer(image, mode="interactive", tool=tool, previous=test_params, max_dim=max_dim)
        elif not df_masks.__class__.__name__ == "NoneType" and flag_edit == True:
            out = _image_viewer(image, mode="interactive", tool=tool, previous=prev_masks, max_dim=max_dim)
        else:
            out = _image_viewer(image, mode="interactive", tool=tool, max_dim=max_dim)
            
        ## abort
        if not out.done:
            if obj_input.__class__.__name__ == "ndarray":
                print("terminated mask creation")
                return
            elif obj_input.__class__.__name__ == "container":
                print("- terminated mask creation")
                return True
        else:
            coords = out.point_list

        ## create df
        df_masks_sub_new = pd.DataFrame()
        if len(coords) > 0:
            for points in coords:
                df_masks_sub_new = df_masks_sub_new.append(
                    {"mask": label, "include": include, "coords": str(points)},
                    ignore_index=True,
                    sort=False,
                )
        else:
            print("zero coordinates - redo mask!")
            break

        ## merge with existing image_data frame
        df_masks_sub_new = pd.concat(
            [
                pd.concat([df_image_data] * len(df_masks_sub_new)).reset_index(drop=True),
                df_masks_sub_new.reset_index(drop=True),
            ],
            sort=False,
            axis=1,
        )
        df_masks = df_masks.append(df_masks_sub_new, sort=False)
        df_masks = df_masks.reindex(df_masks_sub_new.columns, axis=1)

        ## drop index before saving
        df_masks.reset_index(drop=True, inplace=True)
        break

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_masks
    elif obj_input.__class__.__name__ in ["container", "motion_tracker"]:
        obj_input.df_masks = df_masks


def create_reference(
    obj_input,
    df_image_data=None,
    df_masks=None,
    mask=False,
    overwrite=False,
    template=False,
    **kwargs
):
    """
    Measure a size or colour reference card. Minimum input interaction is 
    measuring a size reference: click on two points inside the provided image, 
    and enter the distance - returns the pixel-to-mm-ratio as integer or 
    inserts it into a provided DataFrame (df_image_data). In an optional second
    step, drag a rectangle mask over the reference card to exclude it from any
    subsequent image segementation. The mask is exported as new DataFrame, OR, 
    if provided before, gets appended to an existing one (df_masks). The mask
    can also be stored as a template for automatic reference detection with the
    "detect_reference" function.

    Parameters
    ----------
    obj_input : array or container
        input object
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata to add the reference 
        information to (pixel-to-mm-ratio)
    df_masks : DataFrame, optional
        an existing DataFrame containing masks to add the created mask to
    mask : bool, optional
        mask a reference card inside the image (returns a mask DataFrame)
    overwrite : bool, optional
        if a container is supplied, or when working from the pype, should any 
        exsting reference information (px-to-mm-ratio) or template be overwritten
    template: bool, optional
        should a template for reference detection be created. with an existing 
        template, phenopype can try to find a reference card in a given image,
        measure its dimensions, and adjust and colour space. automatically 
        creates and returns a mask DataFrame that can be added to an existing
        one
    kwargs: optional
        developer options

    Returns
    -------
    px_mm_ratio: int or container
        pixel to mm ratio - not returned if df_image_data is supplied
    df_image_data: DataFrame or container
        new or updated, containes reference information
    df_masks: DataFrame or container
        new or updated, contains mask information
    template: array or container
        template for reference card detection

    """

    ## kwargs
    if df_image_data.__class__.__name__ == "DataFrame":
        flag_df = True
    else:
        flag_df = False
    flag_mask = mask
    flag_template = template
    flag_overwrite = overwrite
    test_params = kwargs.get("test_params", {})

    ## load image
    px_mm_ratio = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame(
                {"filename": "unknown"}, index=[0]
            )  ## may not be necessary
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "reference_template_px_mm_ratio"):
            px_mm_ratio = copy.deepcopy(obj_input.reference_template_px_mm_ratio)
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)
    else:
        print("wrong input format.")
        return

    ## check if exists
    while True:
        if not px_mm_ratio.__class__.__name__ == "NoneType" and flag_overwrite == False:
            print("- pixel-to-mm-ratio already measured (overwrite=False)")
            break
        elif not px_mm_ratio.__class__.__name__ == "NoneType" and flag_overwrite == False:
            print("- measure pixel-to-mm-ratio (overwritten)")
            pass
        elif px_mm_ratio.__class__.__name__ == "NoneType":
            print("- measure pixel-to-mm-ratio")
            pass

        ## method
        out = _image_viewer(image, tool="reference", previous=test_params)

        points = out.reference_coords
        distance_px = sqrt(
                ((points[0][0] - points[1][0]) ** 2)
                + ((points[0][1] - points[1][1]) ** 2)
            )
        
        entry = enter_data(image, columns="length", test_params=test_params)
        distance_mm = float(entry["length"][0])
        px_mm_ratio = float(distance_px / distance_mm)

        ## create template for image registration
        if flag_template or flag_mask:
            out = _image_viewer(image, tool="template", previous=test_params)

            ## make template and mask
            template = image[
                out.rect_list[0][1] : out.rect_list[0][3],
                out.rect_list[0][0] : out.rect_list[0][2],
            ]
            coords = out.point_list

            ## check if exists
            while True:
                if not df_masks.__class__.__name__ == "NoneType":
                    if "reference" in df_masks["mask"].values and flag_overwrite == False:
                        print("- reference template mask already created (overwrite=False)")
                        break
                    elif "reference" in df_masks["mask"].values and flag_overwrite == True:
                        print("- add reference template mask (overwritten)")
                        df_masks = df_masks[~df_masks["mask"].isin(["reference"])]
                        pass

                ## make mask df
                if len(coords) > 0:
                    points = coords[0]
                    df_mask_temp = pd.DataFrame(
                        {"mask": "reference", "include": False, "coords": str(points)},
                        index=[0],
                    )
                    df_mask_temp = pd.concat(
                        [df_image_data, df_mask_temp], axis=1, sort=True
                    )

                    ## add to existing df
                    if df_masks.__class__.__name__ == "NoneType" or len(df_masks) == 0:
                        df_masks = df_mask_temp
                        break
                    if len(df_masks) > 0:
                        df_masks = df_masks.append(df_mask_temp, sort=True)
                        break

                else:
                    print("zero coordinates - redo template!")
                    break
            break
        else:
            template = None
            break

    ## add reference info to data frame
    df_image_data["px_mm_ratio"] = px_mm_ratio

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        if flag_mask:
            if flag_df:
                return df_image_data, df_masks
            else:
                return px_mm_ratio, df_masks
        if flag_template:
            if flag_df:
                return df_image_data, df_masks, template
            else:
                return px_mm_ratio, df_masks, template
        if not flag_template or flag_mask:
            if flag_df:
                return df_image_data
            else:
                return px_mm_ratio
    elif obj_input.__class__.__name__ == "container":
        obj_input.reference_manually_measured_px_mm_ratio = px_mm_ratio
        obj_input.reference_manual_mode = True
        obj_input.df_image_data = df_image_data
        obj_input.df_masks = df_masks
        obj_input.reference_template_image = template



def detect_reference(
    obj_input,
    df_image_data=None,
    template=None,
    overwrite=False,
    equalize=False,
    min_matches=10,
    resize=1,
    px_mm_ratio_ref=None,
    df_masks=None,
):
    """
    Find reference from a template created with "create_reference". Image registration 
    is run by the "AKAZE" algorithm. Future implementations will include more 
    algorithms to select from. First, use "create_reference" with "template=True"
    and pass the template to this function. This happends automatically in the 
    low and high throughput workflow (i.e., when "obj_input" is a container, the 
    template image is contained within. Use "equalize=True" to adjust the 
    histograms of all colour channels to the reference image.
    
    AKAZE: http://www.bmva.org/bmvc/2013/Papers/paper0013/abstract0013.pdf

    Parameters
    -----------
    obj_input: array or container
        input for processing
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata to add the reference 
        information to (pixel-to-mm-ratio)
    df_masks : DataFrame, optional
        an existing DataFrame containing masks to add the detected mask to
    template : array or container, optional
        reference image of reference
    equalize : bool, optional
        should the provided image be colour corrected to match the template 
        images' histogram
    min_matches : int, optional
       minimum key point matches for image registration
    resize: num, optional
        resize image to speed up detection process. default: 0.5 for 
        images with diameter > 5000px (WARNING: too low values may 
        result in poor detection performance or even crashes)
    overwrite : bool, optional
        overwrite existing reference_detected_px_mm_ratio in container
    px_mm_ratio_ref : int, optional
        pixel-to-mm-ratio of the template image

    Returns
    -------
    reference_detected_px_mm_ratio: int or container
        pixel to mm ratio of current image
    image: array or container
        if reference contains colour information, this is the corrected image
    df_masks: DataFrame or container
        contains mask coordinates to mask reference card within image from 
        segmentation algorithms
    """

    ## kwargs
    flag_overwrite = overwrite
    flag_equalize = equalize

    ## load image
    template_px_mm_ratio = None
    template_image = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
        else:
            if "template_px_mm_ratio" in df_image_data:
                template_px_mm_ratio = df_image_data["template_px_mm_ratio"]
                print("template_px_mm_ratio loaded")
        if df_masks.__class__.__name__ == "NoneType":
            df_masks = pd.DataFrame(columns=["mask", "include", "coords"])
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "reference_template_px_mm_ratio"):
            template_px_mm_ratio = obj_input.reference_template_px_mm_ratio
        if hasattr(obj_input, "reference_template_image"):
            template_image = obj_input.reference_template_image
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)
        else:
            df_masks = pd.DataFrame(columns=["mask", "include", "coords"])

    ## check if all info has been provided
    while True:
        if any(
            [
                template_px_mm_ratio.__class__.__name__ == "NoneType",
                template_image.__class__.__name__ == "NoneType",
            ]
        ):
            print("- reference information missing - abort")
            break
        if hasattr(obj_input, "reference_detected_px_mm_ratio") and not flag_overwrite:
            detected_px_mm_ratio = obj_input.reference_detected_px_mm_ratio
            print("- reference already detected (overwrite=False)")
            break
        elif hasattr(obj_input, "reference_detected_px_mm_ratio") and flag_overwrite:
            print(" - detecting reference (overwriting)")
            pass

        ## if image diameter bigger than 5000 px, then automatically resize
        if (image.shape[0] + image.shape[1]) / 2 > 5000 and resize == 1:
            resize_factor = 0.5
            print(
                "large image - resizing by factor "
                + str(resize_factor)
                + " to avoid slow image registration"
            )
        else:
            resize_factor = resize
        image = cv2.resize(image, (0, 0), fx=1 * resize_factor, fy=1 * resize_factor)

        ## method
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(template_image, None)
        kp2, des2 = akaze.detectAndCompute(image, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(des1, des2, 2)

        # keep only good matches
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # find and transpose coordinates of matches
        if len(good) >= min_matches:
            ## find homography betweeen detected keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            ## transform boundary box of template
            rect_old = np.array(
                [
                    [[0, 0]],
                    [[0, template_image.shape[0]]],
                    [[template_image.shape[1], template_image.shape[0]]],
                    [[template_image.shape[1], 0]],
                ],
                dtype=np.float32,
            )
            rect_new = cv2.perspectiveTransform(rect_old, M) / resize_factor

            # calculate template diameter
            rect_new = rect_new.astype(np.int32)
            (x, y), radius = cv2.minEnclosingCircle(rect_new)
            diameter_new = radius * 2

            # calculate transformed diameter
            rect_old = rect_old.astype(np.int32)
            (x, y), radius = cv2.minEnclosingCircle(rect_old)
            diameter_old = radius * 2

            ## calculate ratios
            diameter_ratio = diameter_new / diameter_old
            px_mm_ratio_new = round(diameter_ratio * template_px_mm_ratio, 1)

            ## add to image df
            df_image_data["current_px_mm_ratio"] = px_mm_ratio_new

            ## feedback
            print("---------------------------------------------------")
            print("Reference card found with %d keypoint matches:" % len(good))
            print("template image has %s pixel per mm." % (template_px_mm_ratio))
            print("current image has %s pixel per mm." % (px_mm_ratio_new))
            print("= %s %% of template image." % round(diameter_ratio * 100, 3))
            print("---------------------------------------------------")

            ## create mask from new coordinates
            coords = _contours_arr_tup(rect_new)
            coords.append(coords[0])
            if "reference" in df_masks["mask"].values:
                df_masks = df_masks[~df_masks["mask"].isin(["reference"])]
            row_reference = pd.DataFrame(
                {"mask": "reference", "include": False, "coords": str([coords])}, index=[0]
            )
            row_reference = pd.concat(
                [
                    pd.concat([df_image_data] * len(row_reference)).reset_index(drop=True),
                    row_reference.reset_index(drop=True),
                ],
                axis=1,
            )
            df_masks = df_masks.append(row_reference, sort=False)
            detected_px_mm_ratio = px_mm_ratio_new
            break
        else:
            ## feedback
            print("---------------------------------------------------")
            print("Reference card not found - %d keypoint matches:" % len(good))
            print('Setting "current reference" to None')
            print("---------------------------------------------------")
            detected_px_mm_ratio = None
            break

        ## merge with existing image_data frame
        df_image_data["current_px_mm_ratio"] = detected_px_mm_ratio

    # ## rectangle coords of reference in image
    # rect_new = eval(df_masks.loc[df_masks["mask"]=="reference", "coords"].reset_index(drop=True)[0])

    ## do histogram equalization
    if flag_equalize:
        detected_rect_mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(detected_rect_mask, [np.array(rect_new)], colours["white"])
        (rx, ry, rw, rh) = cv2.boundingRect(np.array(rect_new))
        detected_rect_mask = ma.array(
            data=image[ry : ry + rh, rx : rx + rw],
            mask=detected_rect_mask[ry : ry + rh, rx : rx + rw],
        )
        image = _equalize_histogram(image, detected_rect_mask, template_image)
        print("histograms equalized")

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_image_data, df_masks, image
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_image_data = df_image_data
        obj_input.df_masks = df_masks
        obj_input.reference_detected_px_mm_ratio = detected_px_mm_ratio
        if flag_equalize:
            obj_input.image_copy = image
            obj_input.image = image



def enter_data(
    obj_input,
    df_image_data=None,
    columns="ID",
    overwrite=False,
    label_size="auto",
    label_width="auto",
    label_colour="red",
    **kwargs
):
    """
    Generic data entry that can be added as columns to an existing DataFrame. 
    Useful for images containing labels, or other comments. 

    Parameters
    ----------
    obj_input : array or container
        input object
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata to add columns to
    columns : str or list
        columns to be added to existing or generic data frame
    overwrite : bool, optional
        overwrite existing columns in df
    fontsize : int, optional
        fonsize for onscreen display. 
    font_col : {"red", "green", "blue", "black", "white"} str, optional
        font colour for onscreen display. 

    Returns
    -------
    df_other_data: DataFrame or container
        contains the entered data

    """

    ## kwargs
    flag_overwrite = overwrite
    test_params = kwargs.get("test_params", {})
    flag_test_mode = False
    if "flag_test_mode" in test_params:
        flag_test_mode = test_params["flag_test_mode"]

    ## format columns
    if not columns.__class__.__name__ == "list":
        columns = columns.replace(" ", "")
        columns = columns.split(",")

    ## load image
    df_other_data = pd.DataFrame({}, index=[0])
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        image_copy = copy.deepcopy(obj_input)
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame(
                {"filename": "unknown"}, index=[0]
            )  ## may not be necessary
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image)
        image_copy = copy.deepcopy(obj_input.image)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_other_data"):
            df_other_data = obj_input.df_other_data
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if label_size == "auto":
        label_size = int(_auto_text_size(image) * 3)
    if label_width == "auto":
        label_width = int(_auto_text_width(image) * 3)

    ## keyboard listener
    def _keyboard_entry(event, x, y, flags, params):
        pass

    ## check if exists
    while True:
        for col in columns:
            if col in df_other_data and flag_overwrite == False:
                print("- column " + col + " already created (overwrite=False)")
                continue
            elif col in df_other_data and flag_overwrite == True:
                df_other_data.drop([col], inplace=True, axis=1)
                print("- add column " + col + " (overwriting)")
                pass
            else:
                print("- add column " + col)
                pass

            ## method
            ## while loop keeps opencv window updated when entering data
            entry = ""
            k = 0
            while True or entry == "":

                cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("phenopype", _keyboard_entry)

                if not flag_test_mode:
                    k = cv2.waitKey(1)
                    if k > 0 and k != 8 and k != 13 and k != 27:
                        entry = entry + chr(k)
                    elif k == 8:
                        entry = entry[0 : len(entry) - 1]
                else:
                    entry = test_params["entry"]

                image = copy.deepcopy(image_copy)
                cv2.putText(
                    image,
                    "Enter " + col + ": " + entry,
                    (int(image.shape[0] // 10), int(image.shape[1] / 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_size,
                    colours[label_colour],
                    label_width,
                    cv2.LINE_AA,
                )
                cv2.imshow("phenopype", image)

                if k == 27:
                    cv2.destroyWindow("phenopype")
                    break
                    return True
                elif k == 13:
                    if not entry == "":
                        cv2.destroyWindow("phenopype")
                        break
                elif flag_test_mode:
                    cv2.destroyWindow("phenopype")
                    break
            df_other_data[col] = entry
        break

    ## drop unspecified columns
    for col in list(df_other_data):
        if col not in columns:
            df_other_data.drop(columns=col, inplace=True)

    ## merge with existing image_data frame
    df_image_data = pd.concat(
        [
            pd.concat([df_image_data] * len(df_other_data)).reset_index(drop=True),
            df_other_data.reset_index(drop=True),
        ],
        axis=1,
    )

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_image_data
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_image_data = df_image_data
        obj_input.df_other_data = df_other_data



def invert_image(obj_input):
    """
    Invert all pixel intensities in image (e.g. 0 to 255 or 100 to 155)

    Parameters
    ----------
    obj_input: array or container
        input for processing

    Returns
    -------
    image : array or container
        inverted image
    """

    ## load image and check if pp-project
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image)

    ## method
    image = cv2.bitwise_not(image)

    ## return
    if obj_input.__class__.__name__ == "container":
        obj_input.image = image
    else:
        return image


def resize_image(obj_input, factor=1):
    """
    Resize image by resize factor 

    Parameters
    ----------
    obj_input: array or container
        input for processing
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of 
        original size).

    Returns
    -------
    image : array or container
        resized image

    """

    ## load image and check if pp-project
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image)
        df_image_data = obj_input.df_image_data

    ## method
    image = cv2.resize(
        image, (0, 0), fx=1 * factor, fy=1 * factor, interpolation=cv2.INTER_AREA
    )

    ## return
    if obj_input.__class__.__name__ == "container":
        df_image_data["resized"] = factor
        obj_input.image = image
        obj_input.image_copy = image
        obj_input.df_image_data = df_image_data
    else:
        return image
