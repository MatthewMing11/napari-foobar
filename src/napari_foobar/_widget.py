"""
cellpose dock widget module
"""
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import time
import numpy as np
import logging

from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
import sys

# initialize logger
# use -v or --verbose when starting napari to increase verbosity
logger = logging.getLogger(__name__)
if '--verbose' in sys.argv or '-v' in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)


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

def widget_wrapper():
    from napari.qt.threading import thread_worker
    try:
        from torch import no_grad
    except ImportError:
        def no_grad():
            def _deco(func):
                return func
            return _deco

    @thread_worker
    @no_grad()
    def run_cellpose(image, model_type, channels, channel_axis, diameter,
                    net_avg, resample, 
                    anisotropy,
                    model_match_threshold, do_3D):
        from cellpose import models

        flow_threshold = (31.0 - model_match_threshold) / 10.
        if model_match_threshold==0.0:
            flow_threshold = 0.0
            logger.debug('flow_threshold=0 => no masks thrown out due to model mismatch')
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
    def compute_masks(masks_orig, flows_orig, model_match_threshold):
        import cv2
        from cellpose.utils import fill_holes_and_remove_small_masks
        from cellpose.dynamics import get_masks
        from cellpose.transforms import resize_image

        #print(flows_orig[3].shape, flows_orig[2].shape, masks_orig.shape)
        flow_threshold = (31.0 - model_match_threshold) / 10.
        if model_match_threshold==0.0:
            flow_threshold = 0.0
            logger.debug('flow_threshold=0 => no masks thrown out due to model mismatch')
        logger.debug(f'computing masks with flow_threshold={flow_threshold}')
        maski = get_masks(flows_orig[3].copy(), iscell=(flows_orig[2] > 0.0),
                        flows=flows_orig[1], threshold=flow_threshold*(masks_orig.ndim<3))
        maski = fill_holes_and_remove_small_masks(maski)
        maski = resize_image(maski, masks_orig.shape[-2], masks_orig.shape[-1],
                                        interpolation=cv2.INTER_NEAREST)
        return maski 

    @magicgui(
        call_button='run segmentation',  
        layout='vertical',
        diameter = dict(widget_type='LineEdit', label='diameter', value=30, tooltip='approximate diameter of cells to be segmented'),
        compute_diameter_button  = dict(widget_type='PushButton', text='compute diameter from image', tooltip='cellpose model will estimate diameter from image using specified channels'),
        anisotropy = dict(widget_type='FloatSlider', name='anisotropy', value=1.0, min=1.0, max=10.0, step=0.2, tooltip='resolution ratio between z res to xy res'),
        model_match_threshold = dict(widget_type='FloatSlider', name='model_match_threshold', value=27.0, min=0.0, max=30.0, step=0.2, tooltip='threshold on gradient match to accept a mask (set lower to get more cells)'),
        compute_masks_button  = dict(widget_type='PushButton', text='recompute last masks with new cellprob + model match', enabled=False),
        process_3D = dict(widget_type='CheckBox', text='process stack as 3D', value=False, tooltip='use default 3D processing where flows in X, Y, and Z are computed and dynamics run in 3D to create masks'),
        clear_previous_segmentations = dict(widget_type='CheckBox', text='clear previous results', value=True),
    )
    def widget(#label_logo, 
        viewer: Viewer,
        image_layer: Image,
        diameter,
        compute_diameter_button,
        anisotropy,
        model_match_threshold,
        compute_masks_button,
        process_3D,
        clear_previous_segmentations
    ) -> None:
        # Import when users activate plugin

        if not hasattr(widget, 'cellpose_layers'):
            widget.cellpose_layers = []
        
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
                                model_match_threshold=model_match_threshold,
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


    @widget.compute_masks_button.changed.connect 
    def _compute_masks(e: Any):
        
        mask_worker = compute_masks(widget.masks_orig, 
                                    widget.flows_orig,
                                    widget.model_match_threshold.value)
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
        ### TODO 3D images

    return widget            


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'cellpose'} 