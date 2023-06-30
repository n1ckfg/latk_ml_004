bl_info = {
    "name": "latk-ml-004", 
    "author": "Nick Fox-Gieg",
	"version": (0, 0, 1),
	"blender": (3, 0, 0),
    "description": "Generate brushstrokes from a mesh using informative-drawings",
    "category": "Animation"
}

import bpy
import gpu
import bgl
from bpy.types import Operator, AddonPreferences
from bpy.props import (BoolProperty, FloatProperty, StringProperty, IntProperty, PointerProperty, EnumProperty)
from bpy_extras.io_utils import (ImportHelper, ExportHelper)

import argparse
import cv2
import numpy as np
import onnxruntime

class latkml004Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    '''
    extraFormats_AfterEffects: bpy.props.BoolProperty(
        name = 'After Effects JSX',
        description = "After Effects JSX export",
        default = False
    )
	'''

    def draw(self, context):
        layout = self.layout

        layout.label(text="none")
        #row = layout.row()
        #row.prop(self, "extraFormats_Painter")

# This is needed to display the preferences menu
# https://docs.blender.org/api/current/bpy.types.AddonPreferences.html
class OBJECT_OT_latkml004_prefs(Operator):
    """Display example preferences"""
    bl_idname = "object.latkml004"
    bl_label = "latkml004 Preferences"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        return {'FINISHED'}

class latkml004Properties(bpy.types.PropertyGroup):
    """Properties for latkml004"""
    bl_idname = "GREASE_PENCIL_PT_latkml004Properties"

    '''
    bakeMesh: BoolProperty(
        name="Bake",
        description="Off: major speedup if you're staying in Blender. On: slower but keeps everything exportable",
        default=True
    )
	'''

class latkml004_Button_AllFrames(bpy.types.Operator):
    """Operate on all frames"""
    bl_idname = "latkml004_button.allframes"
    bl_label = "All Frames"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        # function goes here
        pass
        return {'FINISHED'}

class latkml004_Button_SingleFrame(bpy.types.Operator):
    """Operate on a single frame"""
    bl_idname = "latkml004_button.singleframe"
    bl_label = "Single Frame"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        # function goes here
        renderTest()
        return {'FINISHED'}

# https://blender.stackexchange.com/questions/167862/how-to-create-a-button-on-the-n-panel
class latkml004Properties_Panel(bpy.types.Panel):
    """Creates a Panel in the 3D View context"""
    bl_idname = "GREASE_PENCIL_PT_latkml004PropertiesPanel"
    bl_space_type = 'VIEW_3D'
    bl_label = "latkml004"
    bl_category = "Latk"
    bl_region_type = 'UI'
    #bl_context = "objectmode" # "mesh_edit"

    #def draw_header(self, context):
        #self.layout.prop(context.scene.freestyle_gpencil_export, "enable_latk", text="")

    def draw(self, context):
        layout = self.layout

        scene = context.scene
        latkml004 = scene.latkml004_settings

        row = layout.row()
        row.operator("latkml004_button.singleframe")
        row.operator("latkml004_button.allframes")
        #row.prop(latkml004, "material_shader_mode")

classes = (
    OBJECT_OT_latkml004_prefs,
    latkml004Preferences,
    latkml004Properties,
    latkml004Properties_Panel,
	latkml004_Button_AllFrames,
	latkml004_Button_SingleFrame
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)   
    bpy.types.Scene.latkml004_settings = bpy.props.PointerProperty(type=latkml004Properties)

def unregister():
    del bpy.types.Scene.latkml004_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# https://blender.stackexchange.com/questions/262742/python-bpy-2-8-render-directly-to-matrix-array
# https://blender.stackexchange.com/questions/2170/how-to-access-render-result-pixels-from-python-script/3054#3054
def renderTest():
    '''
    w = bpy.context.scene.render.resolution_x
    h = bpy.context.scene.render.resolution_y

    # switch on nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
      
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
      
    # create input render layer node
    rl = tree.nodes.new("CompositorNodeRLayers")      
    rl.location = 185,285
     
    # create output node
    v = tree.nodes.new("CompositorNodeViewer")   
    v.location = 750,210
    v.use_alpha = False
     
    # Links
    links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input
     
    # render
    bpy.ops.render.render()
     
    # get viewer pixels
    image = bpy.data.images["Viewer Node"]
    pixels = image.pixels
    print(len(pixels)) # size is always width * height * 4 (rgba)
     
    # copy buffer to numpy array for faster manipulation
    image_data = np.array(pixels[:])

    # Reshape and flip the image vertically
    image_data = image_data.reshape(w, h, 4)
    image_data = np.flipud(image_data)
    '''
    # Access the pixel values
    '''
    for x in range(0, w):
        for y in range(0, h):
            col = image_data[x, y]
            col = [1, col[1], col[2], col[3]]
    '''

    bpy.ops.render.render()

    render_result = next(image for image in bpy.data.images if image.type == "RENDER_RESULT")

    # Create a GPU texture that shares GPU memory with Blender
    gpu_tex = gpu.texture.from_image(render_result)

    # Read image from GPU
    gpu_tex.read()

    # OR read image into a NumPy array (might be more convenient for later operations)
    fbo = gpu.types.GPUFrameBuffer(color_slots=(gpu_tex,))

    buffer_np = np.empty(gpu_tex.width * gpu_tex.height * 4, dtype=np.float32)
    buffer = bgl.Buffer(bgl.GL_FLOAT, buffer_np.shape, buffer_np)
    with fbo.bind():
        bgl.glReadBuffer(bgl.GL_BACK)
        bgl.glReadPixels(0, 0, gpu_tex.width, gpu_tex.height, bgl.GL_RGBA, bgl.GL_FLOAT, buffer)

    # Now the NumPy array has the pixel data, you can reshape it and/or export it as bytes if you wish
    print(buffer_np)

    render_result.file_format = 'PNG'
    render_result.filepath = "/Users/nick/Desktop/test.png"
    render_result.save()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

'''
@echo off

set STYLE=anime_style
rem STYLE=opensketch_style
set RGB_PATH=input
set DEPTH_PATH=output
set RESULT_PATH=results\%STYLE%
set MAX_FRAMES=999
set RENDER_RES=480

rmdir /s /q %RESULT_PATH%
python informative-drawings\test.py --name %STYLE% --dataroot %RGB_PATH% --how_many %MAX_FRAMES% --size %RENDER_RES%

rmdir /s /q %DEPTH_PATH%
python midas\run.py --input_path %RGB_PATH% --output_path %DEPTH_PATH% --model_weights midas\model\model-f6b98070.pt 

set LINE_THRESHOLD=64
set USE_SWIG=1
set INPAINT=0
set DEPTH_CAMERA_NAME="apple_lidar"
set DEPTH_CAMERA_MODE="default"

python skeletonizer.py -- %RESULT_PATH% %RGB_PATH% %DEPTH_PATH% %LINE_THRESHOLD% %USE_SWIG% %INPAINT% %DEPTH_CAMERA_NAME% %DEPTH_CAMERA_MODE%

@pause
'''

class Informative_Drawings():
    def __init__(self, modelpath):
        try:
            cv_net = cv2.dnn.readNet(modelpath)
        except:
            print('opencv read onnx failed!!!')
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(modelpath, so)
        input_shape = self.net.get_inputs()[0].shape
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(np.transpose(img.astype(np.float32), (2, 0, 1)), axis=0).astype(np.float32)
        outs = self.net.run([self.output_name], {self.input_name: blob})

        result = outs[0].squeeze()
        result *= 255
        result = cv2.resize(result.astype('uint8'), (srcimg.shape[1], srcimg.shape[0]))
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/2.jpg', help='image path')
    parser.add_argument("--modelpath", type=str, default='weights/opensketch_style_512x512.onnx', choices=["weights/opensketch_style_512x512.onnx", "weights/anime_style_512x512.onnx", "weights/contour_style_512x512.onnx"], help='onnx filepath')
    args = parser.parse_args()

    mynet = Informative_Drawings(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    result = mynet.detect(srcimg)

    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', srcimg)
    winName = 'Deep learning in onnxruntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
