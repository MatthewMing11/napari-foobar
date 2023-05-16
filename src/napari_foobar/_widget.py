"""
cellpose dock widget module
copied from https://github.com/MouseLand/cellpose-napari/blob/main/cellpose_napari/_dock_widget.py
"""
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import time
import numpy as np
import logging
from skimage import data
from skimage.segmentation import clear_border
from skimage.measure import label

from napari import Viewer
from napari.layers import Image, Shapes, Labels
from magicgui import magicgui
import sys

##LOGGING
# initialize logger
# use -v or --verbose when starting napari to increase verbosity
logger = logging.getLogger(__name__)
if '--verbose' in sys.argv or '-v' in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)

#filtering functions for cell analysis
def getCells(labels):
        data = labels
        shape = data.shape #shape is in [z,y,x] form

        # get data points("cell_ids") from data 
        cell_ids = dict()
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    cell_id = data[z][y][x]
                    if cell_id:     #first checks if value is valid to put into cell_ids(0 is empty space)
                        if cell_id not in cell_ids:     #if cell_id isn't in cell_ids, instantiate empty list
                            cell_ids.setdefault(cell_id,[])
                        cell_ids[cell_id].append([x,y,z]) #append data point to the cell_id list
        return cell_ids

def generate_volume(cell_ids):
        """Return number of pixels("volume") each cell_id is made of to get cell volume"""
        volumes=dict()
        for key in cell_ids.keys():
            volumes[key]=len(cell_ids[key])
        return volumes

def getTop(cell_ids):
    sizes=generate_volume(cell_ids)
    listofsizes=list(reversed(sorted(sizes,key=sizes.get)))
    return listofsizes

def generate_centers(cell_ids):
    """Return centers(heavily relies on distribution of points) of each cell_id"""
    centers=dict()
    for key in cell_ids.keys():
        centers[key]=[round(sum(col)/len(cell_ids[key])) for col in zip(*cell_ids[key])]
    return centers
#filter functions end here

#merge functions start here
def mergeCells(labels):
    import cv2 as cv
    orig=labels
    # array=np.copy(orig)
    # array=[]
    im2=np.copy(orig)
    for index in range(len(orig)): #go through all slices
        print("Index")
        print(index)
        print(f'dtype: {orig.dtype}, shape: {orig.shape}, min: {np.min(orig)}, max: {np.max(orig)}')#check contents of file(not metadata)
        # print(orig[0,:,:].shape)
        # slice=orig[130,:,:]#take 130th slice of the image(testing purposes)
        slice=im2[index,:,:]
        prevslice=np.array([])
        nextslice=np.array([])
        if index:
            prevslice=im2[index-1,:,:]
        if index+1!=len(orig):
            nextslice=im2[index+1,:,:]
        im= np.uint8(slice)#conversion because i need to get the sobel gradient to get contours
        
        # compute the 1st order Sobel derivative in X-direction
        sobelx = cv.Sobel(im,cv.CV_64F,1,0,ksize=3)
        # compute the 1st order Sobel derivative in Y-direction
        sobely = cv.Sobel(im,cv.CV_64F,0,1,ksize=3)
        #computes gradient given the Sobel derivatives
        grad=np.sqrt(np.square(sobelx)+np.square(sobely))
        grad=grad.astype('uint8')
        h, w = im.shape[:2]
        contours, hierarchy = cv.findContours(grad, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)#get contours of gradient(helps grab all inner contours)
        # I am taking only those contours which do not have a child contour.
        # finalContours = np.asarray([contours[i] for i in range(len(contours)) if hierarchy[0][i][2] == -1])
        # Iterating over each pixel except those at the boundary
        contours=np.asarray(contours).tolist()
        listofContours=[]
        #Remake im with only inner contours in new variable edges. Record this list of contours for visualization later.
        edge = np.zeros(im.shape, dtype = np.uint8)
        for i in contours:
            row=[]
            for j in i:
                for k in j:
                    arr=k.tolist()
                    x=arr[0]
                    y=arr[1]
                    if x>0 and x<h and y>1 and y<w:# check if within bounds of image
                        roi = slice[y-1:y+2, x-1:x+2].copy()
                        roi2=np.array([])
                        roi3=np.array([])
                        if np.any(prevslice):
                            roi2 = prevslice[y-1:y+2, x-1:x+2].copy()
                        if np.any(nextslice):
                            roi3 = nextslice[y-1:y+2, x-1:x+2].copy()
                        roi[1,1]=0
                        # print("ROI HERE")
                        # print(roi)
                        unique_neighbors=set(np.concatenate((roi.flatten(),roi2.flatten(),roi3.flatten())))
                        # print(unique_neighbors)
                        #keep all non-zero values
                        if 0 in unique_neighbors:
                            unique_neighbors.remove(0)
                        # if contour pixel only borders one cell(itself), remove it
                        # print(unique_neighbors)
                        if len(unique_neighbors)>1:
                            row.append([arr])
                            edge[y][x] = 255
            if len(row)!=0:
                row=np.asarray(row,dtype=np.int32)
                listofContours.append(row)

        # # listofContours=np.array(listofContours)
        # # listofContours = listofContours.astype('int32')
        print(listofContours)
        # #need to make list of Contours look like contours
        # # so far the numpy appending is messing up everything, need to figure out why
        print("Real")
        print(contours)
        print(grad)
        #Go through all contours in list and merge neighboring cells.
        # im2=np.copy(slice)
        for i in listofContours:
            for j in i:
                for k in j:
                    arr=k.tolist()
                    x=arr[0]
                    y=arr[1]
                    roi = slice[y-1:y+2, x-1:x+2].copy()
                    roi2=np.array([])
                    roi3=np.array([])
                    if np.any(prevslice):
                        roi2 = prevslice[y-1:y+2, x-1:x+2].copy()
                    if np.any(nextslice):
                        roi3 = nextslice[y-1:y+2, x-1:x+2].copy()
                    roi[1,1]=0
                    # print(roi)
                    unique_neighbors=set(np.concatenate((roi.flatten(),roi2.flatten(),roi3.flatten())))
                    # print(unique_neighbors)
                    #keep all non-zero values
                    if 0 in unique_neighbors:
                        unique_neighbors.remove(0)
                    if len(unique_neighbors)>1:
                        unique_neighbors=list(unique_neighbors)
                        replace=unique_neighbors[0]
                        for unique_neighbor in unique_neighbors[1:]:
                            if unique_neighbor in im2:#hopefully this speeds up computation
                                print("Executed")
                                im2=np.where(im2==unique_neighbor,replace,im2)
                                print("Done")
    return im2
# Python3 implementation of the approach
 
# Function that returns true if
# the given pixel is valid
def isValid(screen, m, n, x, y, prevC, newC):
    if x<0 or x>= m\
       or y<0 or y>= n or\
       screen[x][y]!= prevC\
       or screen[x][y]== newC:
        return False
    return True
 
 
# FloodFill function
def floodFill(screen, 
            m, n, x, 
            y, prevC, newC):
    queue = []
     
    # Append the position of starting
    # pixel of the component
    queue.append([x, y])
 
    # Color the pixel with the new color
    screen[x][y] = newC
 
    # While the queue is not empty i.e. the
    # whole component having prevC color
    # is not colored with newC color
    while queue:
         
        # Dequeue the front node
        currPixel = queue.pop()
         
        posX = currPixel[0]
        posY = currPixel[1]
         
        # Check if the adjacent
        # pixels are valid
        if isValid(screen, m, n, 
                posX + 1, posY, 
                        prevC, newC):
             
            # Color with newC
            # if valid and enqueue
            screen[posX + 1][posY] = newC
            queue.append([posX + 1, posY])
         
        if isValid(screen, m, n, 
                    posX-1, posY, 
                        prevC, newC):
            screen[posX-1][posY]= newC
            queue.append([posX-1, posY])
         
        if isValid(screen, m, n, 
                posX, posY + 1, 
                        prevC, newC):
            screen[posX][posY + 1]= newC
            queue.append([posX, posY + 1])
         
        if isValid(screen, m, n, 
                    posX, posY-1, 
                        prevC, newC):
            screen[posX][posY-1]= newC
            queue.append([posX, posY-1])

# Adapted from  Binary fill algorithm by Gabriel Landini, G.Landini at bham.ac.uk in ImageJ
def fill_holes(labels):
    from skimage import (
        color, data, filters, measure,
        morphology, segmentation, util
        )
    im=labels
    im2=np.copy(im)
    h, w = im2.shape[1:]
    for index in range(len(im2)):
        print(index)
        for i in range(h):
            if im2[index][0,i]==0: floodFill(im2[index],w,h,0,i,0,127)
            if im2[index][w-1,i]==0: floodFill(im2[index],w,h,w-1,i,0,127)
        for i in range(w):
            if im2[index][i,0]==0: floodFill(im2[index],w,h,i,0,0,127)
            if im2[index][i,h-1]==0: floodFill(im2[index],w,h,i,h-1,0,127)
        n=h*w
        for i in range(n):
            print(i/w)
            if im2[index][int(i/w)][i%w]==127:
                im2[index][int(i/w)][i%w]=0
            else:
                im2[index][int(i/w)][i%w]=255
    binary_im2=im2
    im2 = segmentation.watershed(binary_im2, im, mask=binary_im2)
    return im2
#merge functions end here

#@thread_worker
def read_logging(log_file, logwindow):
    with open(log_file, 'r') as thefile:
        #thefile.seek(0,2) # Go to the end of the file
        while True:
            line = thefile.readline()
            if not line:
                time.sleep(0.01) # Sleep briefly
                continue
            else:
                logwindow.cursor.movePosition(logwindow.cursor.End)
                logwindow.cursor.insertText(line)
                yield line
            
cp_strings = ['_cp_masks_', '_cp_outlines_', '_cp_flows_', '_cp_cellprob_']

#logo = os.path.join(__file__, 'logo/logo_small.png')
##WRAPPER FUNCTION FOR WIDGET
def widget_wrapper():
    from napari.qt.threading import thread_worker
    try:
        from torch import no_grad
    except ImportError:
        def no_grad():
            def _deco(func):
                return func
            return _deco
    ##FUNCTIONS TO BE MULTITHREADED
    @thread_worker
    @no_grad()
    def run_cellpose(image, model_type, channels, channel_axis, diameter,
                    net_avg, resample, 
                    anisotropy,
                    min_size, do_3D):
        from cellpose import models
        import tifffile as tiff

        flow_threshold = 0.4
        logger.debug(f'computing masks with flow_threshold={flow_threshold}')
        CP = models.CellposeModel(model_type=model_type, gpu=True)
        masks, flows_orig, _ = CP.eval(image, 
                                    channels=channels, 
                                    channel_axis=channel_axis,
                                    diameter=diameter,
                                    net_avg=net_avg,
                                    resample=resample,
                                    flow_threshold=flow_threshold,
                                    do_3D=do_3D,
                                    anisotropy=anisotropy,
                                    min_size=min_size,
                                    stitch_threshold=0)
        del CP
        if not do_3D and masks.ndim > 2:
            flows = [[flows_orig[0][i], 
                      flows_orig[1][:,i],
                      flows_orig[2][i],
                      flows_orig[3][:,i]] for i in range(masks.shape[0])]
            masks = list(masks)
            flows_orig = flows
        return masks, flows_orig

    @thread_worker
    def compute_diameter(image, channels, model_type):
        from cellpose import models

        CP = models.Cellpose(model_type = model_type, gpu=True)
        diam = CP.sz.eval(image, channels=channels, channel_axis=-1)[0]
        diam = np.around(diam, 2)
        del CP
        return diam
    @thread_worker
    def resize(labels):
        """Return labels with cells on the border removed"""
        ratio=[widget.pixelZ.value,widget.pixelY.value,widget.pixelX.value]
        ratio=[float(i) for i in ratio]
        labels.scale=[coord//float(widget.pixelX.value) for coord in ratio]
        return labels.data
    @thread_worker
    def merge(labels):
        import tifffile as tiff
        """Merges cells and saves them to a file"""
        masks=mergeCells(labels)
        masks=fill_holes(masks)
        #save after merge
        tiff.imwrite(widget.image_layer.value.name+"_cp_orig.tif",masks)
        return masks
    
    @thread_worker
    def delete_edge(labels):
        """Return labels with cells on the border removed"""
        return clear_border(labels)
    
    # @thread_worker
    def isolate(labels):
        """Isolates all cells so that only one may appear on screen.
            The largest cell is displayed first.
            Press the isolate button on startup of plugin."""
        cells=getCells(labels)
        ordered=getTop(cells)
        centers=generate_centers(cells)
        blank=np.zeros(labels.shape)
        for key, value in centers.items():
            z,y,x=value
            for i in range(-1,2):
                for j in range(-1,2):
                    for k in range(-1,2):
                        blank[x+i,y+j,z+k]=key
        widget.labeled_cells=ordered.copy()
        print(widget.labeled_cells)
        widget.label_layer.value.selected_label=widget.labeled_cells[0]
        widget.label_layer.value.show_selected_label=True
        return blank.astype(labels.dtype)
    
    @thread_worker
    def erode(labels):
        """Applies erosion to current cell on the screen. 
        A counter on the erosion layer shows how many times erosion has been done."""
        import cv2 as cv
        from skimage import (
        measure,segmentation,
        )
        import tifffile as tiff
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max

        kernel = np.ones((3, 3), np.uint8)
        label_isolated=np.where(labels==widget.labeled_cells[widget.current_index],labels,0)
        labels_eroded=cv.erode(np.float32(label_isolated),kernel,iterations=1)
        other_labels=np.where(labels!=widget.labeled_cells[widget.current_index],labels,0)
        dist = ndi.distance_transform_edt(labels_eroded) #make distance map
        coords = peak_local_max(dist,footprint=np.ones((50, 50,50)), labels=labels_eroded.astype('int32'))#.astype('int64'))
        mask = np.zeros(dist.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask,structure=ndi.generate_binary_structure(3, 3))#3d-image(26) connectivity
        output=segmentation.watershed(-dist,markers=markers,mask=labels_eroded)
        # tiff.imwrite("answer.tif",output,bigtiff=True)
        cells=getCells(output)
        new_centers=generate_centers(cells)
        print(len(cells))
        blank=np.zeros(labels.shape)
        for key, value in new_centers.items():
            z,y,x=value
            print("CENTER")
            print(value)
            for i in range(-1,2):#not sure if this is slow or not
                for j in range(-1,2):
                    for k in range(-1,2):
                        blank[x+i,y+j,z+k]=key
        widget.viewer.value.layers["erosion_"+str(widget.erode_count)].name = "erosion_"+str(widget.erode_count+1)
        widget.erode_count += 1
        all_centers=widget.viewer.value.layers["centroids"].data
        other_centers=np.where(all_centers==widget.labeled_cells[widget.current_index],0,all_centers)
        blank=np.where(blank!=0,widget.labeled_cells[widget.current_index],blank)
        blank=blank+other_centers
        print("Erosion done")
        return (labels_eroded+other_labels).astype(labels.dtype),blank.astype(labels.dtype)

    # @thread_worker
    def watershed(labels):
        """Applies watershed to current cell on screen. Once applied, moves to the next largest cell."""
        from skimage import (
        measure,segmentation,
        )
        import tifffile as tiff
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        """Takes a labeled image and finds peaks. Using these peaks, it """
        label_isolated=np.where(labels==widget.labeled_cells[widget.current_index],labels,0)
        other_labels=np.where(labels!=widget.labeled_cells[widget.current_index],labels,0)
        cell_max=max(widget.labeled_cells)
        dist = ndi.distance_transform_edt(label_isolated) #make distance map
        coords = peak_local_max(dist, labels=label_isolated)
        mask = np.zeros(dist.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask,structure=ndi.generate_binary_structure(3, 3))#3d-image(26) connectivity
        output=segmentation.watershed(-dist,markers=markers,mask=label_isolated)
        # tiff.imwrite("answer.tif",np.where(output==0,0,output+cell_max)+other_labels,bigtiff=True)
        print(output+other_labels)
        output=np.where(output==0,0,output+cell_max)+other_labels
        isolate(output)
        widget.viewer.value.layers["erosion_"+str(widget.erode_count)].name = "erosion_0"
        widget.erode_count = 0
        return output
    @thread_worker
    def delete(labels):
        """Deletes the current cell on screen and moves to the next largest cell"""
        labels=np.where(labels!=widget.labeled_cells[widget.current_index],labels,0)
        widget.labeled_cells.pop(widget.current_index)
        widget.label_layer.value.selected_label=widget.labeled_cells[widget.current_index]
        widget.viewer.value.layers["centroids"].selected_label=widget.labeled_cells[widget.current_index]
        widget.viewer.value.layers["erosion_"+str(widget.erode_count)].name = "erosion_0"
        widget.erode_count = 0
        print(widget.labeled_cells)
        return labels
    @thread_worker
    def next(labels):
        """Moves to the next largest cell"""
        if  widget.current_index !=len(widget.labeled_cells)-1:
            widget.current_index += 1
        widget.label_layer.value.selected_label=widget.labeled_cells[widget.current_index]
        widget.viewer.value.layers["centroids"].selected_label=widget.labeled_cells[widget.current_index]
        widget.viewer.value.layers["erosion_"+str(widget.erode_count)].name = "erosion_0"
        widget.erode_count = 0
        print(widget.labeled_cells)
        return labels
    @thread_worker
    def prev(labels):
        """Moves to the previous largest cell"""
        if not widget.current_index:
            widget.current_index -= 1
        widget.label_layer.value.selected_label=widget.labeled_cells[widget.current_index]
        widget.viewer.value.layers["centroids"].selected_label=widget.labeled_cells[widget.current_index]
        widget.viewer.value.layers["erosion_"+str(widget.erode_count)].name = "erosion_0"
        widget.erode_count = 0
        print(widget.labeled_cells)
        return labels
    @thread_worker 
    def compute_masks(masks_orig, flows_orig):
        import cv2
        from cellpose.utils import fill_holes_and_remove_small_masks
        from cellpose.dynamics import get_masks
        from cellpose.transforms import resize_image

        #print(flows_orig[3].shape, flows_orig[2].shape, masks_orig.shape)
        flow_threshold = 0.4
        logger.debug(f'computing masks with flow_threshold={flow_threshold}')
        maski = get_masks(flows_orig[3].copy(), iscell=(flows_orig[2] > 0.0),
                        flows=flows_orig[1], threshold=flow_threshold*(masks_orig.ndim<3))
        maski = fill_holes_and_remove_small_masks(maski)
        maski = resize_image(maski, masks_orig.shape[-2], masks_orig.shape[-1],
                                        interpolation=cv2.INTER_NEAREST)
        return maski 
    ##GUI ELEMENT CREATION
    @magicgui(
        call_button='run segmentation',  
        layout='vertical',
        diameter = dict(widget_type='LineEdit', label='diameter', value=30, tooltip='approximate diameter of cells to be segmented'),
        compute_diameter_button  = dict(widget_type='PushButton', text='compute diameter from image', tooltip='cellpose model will estimate diameter from image using specified channels'),
        anisotropy = dict(widget_type='FloatSlider', name='anisotropy', value=1.0, min=1.0, max=10.0, step=0.2, tooltip='resolution ratio between z res to xy res'),
        min_size = dict(widget_type='Slider', name='min_size', value=15, min=15, max=1000, step=1, tooltip='minimum cell size threshold for cellpose'),
        compute_masks_button  = dict(widget_type='PushButton', text='recompute last masks with new cellprob + model match', enabled=False),
        pixelX = dict(widget_type='LineEdit', label='pixelX', value=1, tooltip='approximate pixel width (in microns)'),
        pixelY = dict(widget_type='LineEdit', label='pixelY', value=1, tooltip='approximate pixel height (in microns)'),
        pixelZ = dict(widget_type='LineEdit', label='pixelZ', value=1, tooltip='approximate pixel depth (in microns)'),
        resize_cells_button= dict(widget_type='PushButton', text='resize cells from image', tooltip='Resize cells from image'),
        process_3D = dict(widget_type='CheckBox', text='process stack as 3D', value=True, tooltip='use default 3D processing where flows in X, Y, and Z are computed and dynamics run in 3D to create masks'),
        clear_previous_segmentations = dict(widget_type='CheckBox', text='clear previous results', value=True),
        merge_cells_button = dict(widget_type='PushButton', text='merge cells from image', tooltip='Merge cells from image using specified channels'),
        delete_edge_button = dict(widget_type='PushButton', text='delete edge cells from image', tooltip='Remove edge cells from image using specified channels'),
        isolate_button = dict(widget_type='PushButton', text='isolate cells from image', tooltip='Isolate cells from image using specified channels'),
        erode_button = dict(widget_type='PushButton', text='erode cells from image', tooltip='Erode cells from image using specified channels'),
        watershed_button = dict(widget_type='PushButton', text='watershed cells from image', tooltip='Watershed cells from image using specified channels'),
        delete_button = dict(widget_type='PushButton', text='delete cells from image', tooltip='Remove cells from image using specified channels'),
        prev_button = dict(widget_type='PushButton', text='prev cell to relabel', tooltip='Moves to previous cell'),
        next_button = dict(widget_type='PushButton', text='next cell to relabel', tooltip='Moves to next cell'),
    )
    #THE WIDGET/PLUGIN
    def widget(#label_logo, 
        viewer: Viewer,
        image_layer: Image,
        label_layer: Labels,
        diameter,
        compute_diameter_button,
        anisotropy,
        min_size,
        compute_masks_button,
        pixelX,
        pixelY,
        pixelZ,
        resize_cells_button,
        process_3D,
        clear_previous_segmentations,
        merge_cells_button,
        delete_edge_button,
        isolate_button,
        erode_button,
        watershed_button,
        delete_button,
        prev_button,
        next_button,
    ) -> None:
        # Import when users activate plugin
        #these variables don't exist in the plugin so make them exist
        if not hasattr(widget, 'cellpose_layers'):
            widget.cellpose_layers = []
        if not hasattr(widget, 'labeled_cells'):
            widget.labeled_cells = []
        if not hasattr(widget, 'erode_count'):
            widget.erode_count = 0
        if not hasattr(widget, 'current_index'):
            widget.current_index = 0
        if clear_previous_segmentations:
            layer_names = [layer.name for layer in viewer.layers]
            for layer_name in layer_names:
                if any([cp_string in layer_name for cp_string in cp_strings]):
                    viewer.layers.remove(viewer.layers[layer_name])
            widget.cellpose_layers = []

        def _new_layers(masks, flows_orig):
            from cellpose.utils import masks_to_outlines
            from cellpose.transforms import resize_image
            import cv2

            flows = resize_image(flows_orig[0], masks.shape[-2], masks.shape[-1],
                                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            cellprob = resize_image(flows_orig[2], masks.shape[-2], masks.shape[-1],
                                    no_channels=True)
            cellprob = cellprob.squeeze()
            outlines = masks_to_outlines(masks) * masks  
            if masks.ndim==3 and widget.n_channels > 0:
                masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
                outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis), 
                                    widget.n_channels, axis=widget.channel_axis)
                flows = np.repeat(np.expand_dims(flows, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
                cellprob = np.repeat(np.expand_dims(cellprob, axis=widget.channel_axis), 
                                    widget.n_channels, axis=widget.channel_axis)
                
            widget.flows_orig = flows_orig
            widget.masks_orig = masks
            widget.iseg = '_' + '%03d'%len(widget.cellpose_layers)
            layers = []

            # get physical scale (...ZYX)
            if len(image_layer.scale) > 3:
                physical_scale = image_layer.scale[-3:]
            else:
                physical_scale = image_layer.scale
            
            layers.append(viewer.add_labels(masks, name=image_layer.name + '_cp_masks' + widget.iseg, visible=False, scale=physical_scale))
            widget.cellpose_layers.append(layers)

        def _new_segmentation(segmentation):
            masks, flows_orig = segmentation
            try:
                if image_layer.ndim > 2 and not process_3D:
                    for mask, flow_orig in zip(masks, flows_orig):
                        _new_layers(mask, flow_orig)
                else:
                    _new_layers(masks, flows_orig)
                
                for layer in viewer.layers:
                    layer.visible = False
                viewer.layers[-1].visible = True
                image_layer.visible = True
                widget.compute_masks_button.enabled = True
                widget.delete_edge_button.enabled = True      
            except Exception as e:
                logger.error(e)
            widget.call_button.enabled = True
            
        image = image_layer.data 
        # put channels last
        widget.n_channels = 0
        widget.channel_axis = None
        if image_layer.ndim == 4 and not image_layer.rgb:
            chan = np.nonzero([a=='c' for a in viewer.dims.axis_labels])[0]
            if len(chan) > 0:
                chan = chan[0]
                widget.channel_axis = chan
                widget.n_channels = image.shape[chan]
        elif image_layer.ndim==3 and not image_layer.rgb:
            image = image[:,:,:,np.newaxis]
        elif image_layer.rgb:
            widget.channel_axis = -1

        cp_worker = run_cellpose(image=image,
                                model_type='nuclei',
                                channels=[0, 0],
                                channel_axis=widget.channel_axis, 
                                diameter=float(diameter),
                                net_avg=True,
                                resample=False,
                                anisotropy=anisotropy,
                                min_size=min_size,
                                do_3D=(process_3D and image_layer.ndim>2),)
        cp_worker.returned.connect(_new_segmentation)
        cp_worker.start()


    def update_masks(masks):     
        from cellpose.utils import masks_to_outlines

        outlines = masks_to_outlines(masks) * masks
        if masks.ndim==3 and widget.n_channels > 0:
            masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis), 
                            widget.n_channels, axis=widget.channel_axis)
            outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
        
        widget.viewer.value.layers[widget.image_layer.value.name + '_cp_masks' + widget.iseg].data = masks
        outline_str = widget.image_layer.value.name + '_cp_outlines' + widget.iseg
        if outline_str in widget.viewer.value.layers:
            widget.viewer.value.layers[outline_str].data = outlines
        widget.masks_orig = masks
        logger.debug('masks updated')

    # I wrote these functions
    def update_layer(masks):
        """Updates the image layer, meant to be used for cellpose functions"""
        widget.viewer.value.layers[widget.image_layer.value.name].data = masks
        # layers=[]
        # layers.append(widget.viewer.add_labels(masks, name=widget.image_layer.name + '_delete' + widget.iseg, visible=False))
        # widget.cellpose_layers.append(layers)

    def update_labels(masks):
        """Updates the label layer"""
        widget.viewer.value.layers[widget.label_layer.value.name].data = masks
    
    def update_centers(masks):
        """Updates the centroid layer and the invisible erosion layer"""
        widget.viewer.value.layers["erosion_"+str(widget.erode_count)].data = masks[0]
        widget.viewer.value.layers["centroids"].data=masks[1]

    #MULTITHREADING FUNCTIONS for respective buttons in GUI
    @widget.compute_masks_button.changed.connect 
    def _compute_masks(e: Any):
        
        mask_worker = compute_masks(widget.masks_orig, 
                                    widget.flows_orig)
        logger.debug(mask_worker)
        mask_worker.returned.connect(update_masks)
        mask_worker.start()

    def _report_diameter(diam):
        widget.diameter.value = diam
        logger.debug(f'computed diameter = {diam}')
    
    @widget.compute_diameter_button.changed.connect 
    def _compute_diameter(e: Any):

        model_type = 'nuclei'
        channels = [0, 0],
        image = widget.image_layer.value.data
        diam_worker = compute_diameter(image, channels, model_type)
        diam_worker.returned.connect(_report_diameter)
        diam_worker.start()

    # I wrote these functions, they connect the functions to the respective buttons
    @widget.resize_cells_button.changed.connect
    def _resize(e:Any):
        image = widget.image_layer.value
        resize_worker = resize(image)
        resize_worker.returned.connect(update_layer)
        resize_worker.start()
    @widget.merge_cells_button.changed.connect
    def _merge(e:Any):
        image = widget.image_layer.value.data
        merge_worker = merge(image)
        merge_worker.returned.connect(update_layer)
        merge_worker.start()

    @widget.delete_edge_button.changed.connect
    def _delete_edge(e:Any):
        image = widget.image_layer.value.data
        delete_worker = delete_edge(image)
        delete_worker.returned.connect(update_layer)
        delete_worker.start()

    @widget.isolate_button.changed.connect
    def _isolate(e:Any):
        image = widget.label_layer.value.data
        scale = widget.label_layer.value.scale
        if not hasattr(widget, 'erode_count'):#it wasn't working on plugin startup so i'll just put it here and it works
            widget.erode_count = 0
        if not hasattr(widget, 'current_index'):
            widget.current_index = 0
        centers=isolate(image)#the only one without a worker because i need the cells to be labeled first
        widget.viewer.value.add_labels(centers,name="centroids",blending="additive",scale=scale)
        widget.viewer.value.add_labels(image,name="erosion_"+str(widget.erode_count),scale=scale)#this is the erosion layer
        widget.viewer.value.layers["centroids"].selected_label=widget.labeled_cells[0]
        widget.viewer.value.layers["centroids"].show_selected_label=True
        widget.viewer.value.layers["erosion_"+str(widget.erode_count)].visible = False
        
    @widget.erode_button.changed.connect
    def _erode(e:Any):
        image = widget.viewer.value.layers["erosion_"+str(widget.erode_count)].data
        erode_worker = erode(image)
        erode_worker.returned.connect(update_centers)
        erode_worker.start()
           
    @widget.watershed_button.changed.connect
    def _watershed(e:Any):#also is not multithreaded like _isolate
        image = widget.label_layer.value.data
        # watershed_worker = watershed(image)
        update_labels(watershed(image))
        widget.label_layer.value.selected_label=widget.labeled_cells[widget.current_index]
        widget.viewer.value.layers["centroids"].selected_label=widget.labeled_cells[widget.current_index]
        # watershed_worker.returned.connect(update_labels)
        # watershed_worker.start()   

    @widget.delete_button.changed.connect
    def _delete(e:Any):
        image = widget.label_layer.value.data
        delete_worker = delete(image)
        delete_worker.returned.connect(update_labels)
        delete_worker.start()   

    @widget.next_button.changed.connect
    def _prev(e:Any):
        image = widget.label_layer.value.data
        prev_worker = prev(image)
        prev_worker.returned.connect(update_labels)
        prev_worker.start()   
    
    @widget.next_button.changed.connect
    def _next(e:Any):
        image = widget.label_layer.value.data
        next_worker = next(image)
        next_worker.returned.connect(update_labels)
        next_worker.start()   
    return widget  

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'cellpose'} 