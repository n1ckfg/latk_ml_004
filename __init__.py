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
import addon_utils

import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime
import latk
import latk_blender as lb
from skimage.morphology import skeletonize
from mathutils import Vector, Quaternion

def findAddonPath(name=None):
    if not name:
        name = __name__
    for mod in addon_utils.modules():
        if mod.bl_info["name"] == name:
            url = mod.__file__
            return os.path.dirname(url)
    return None

sys.path.append(os.path.join(findAddonPath(), "skeleton-tracing/swig"))
from trace_skeleton import *

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

    latkml004_ModelStyle: EnumProperty(
        name="Style",
        items=(
            ("ANIME", "Anime Style", "...", 0),
            ("CONTOUR", "Contour Style", "...", 1),
            ("OPENSKETCH", "OpenSketch Style", "...", 2),
            ("EXPERIMENTAL", "Experimental", "...", 3)
        ),
        default="ANIME"
    )
    
    latkml004_ModelType: EnumProperty(
        name="Type",
        items=(
            ("INFORMATIVE_DRAWINGS", "informative-drawings", "...", 0),
            ("PIX2PIX", "pix2pix", "...", 1)
        ),
        default="INFORMATIVE_DRAWINGS"
    )

    latkml004_lineThreshold: FloatProperty(
        name="lineThreshold",
        description="...",
        default=32.0 # 64
    )

    latkml004_distThreshold: FloatProperty(
        name="distThreshold",
        description="...",
        default=0.5
    )

    latkml004_csize: IntProperty(
        name="csize",
        description="...",
        default=10
    )

    latkml004_maxIter: IntProperty(
        name="iter",
        description="...",
        default=999
    )

    latkml004_thickness: FloatProperty(
        name="thickness",
        description="...",
        default=20.0
    )

class latkml004_Button_AllFrames(bpy.types.Operator):
    """Operate on all frames"""
    bl_idname = "latkml004_button.allframes"
    bl_label = "All Frames"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml004 = context.scene.latkml004_settings
        onnx = loadModel()

        la = latk.Latk()
        la.layers.append(latk.LatkLayer())

        start, end = lb.getStartEnd()
        for i in range(start, end):
            lb.goToFrame(i)
            laFrame = doInference(onnx)
            la.layers[0].frames.append(laFrame)

        lb.fromLatkToGp(la, resizeTimeline=False)
        setThickness(latkml004.latkml004_thickness)
        return {'FINISHED'}

class latkml004_Button_SingleFrame(bpy.types.Operator):
    """Operate on a single frame"""
    bl_idname = "latkml004_button.singleframe"
    bl_label = "Single Frame"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        latkml004 = context.scene.latkml004_settings
        onnx = loadModel()

        la = latk.Latk()
        la.layers.append(latk.LatkLayer())
        laFrame = doInference(onnx)
        la.layers[0].frames.append(laFrame)
        
        lb.fromLatkToGp(la, resizeTimeline=False)
        setThickness(latkml004.latkml004_thickness)
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
        latkml004 = context.scene.latkml004_settings

        layout = self.layout

        row = layout.row()
        row.operator("latkml004_button.singleframe")
        row.operator("latkml004_button.allframes")

        row = layout.row()
        row.prop(latkml004, "latkml004_ModelStyle")

        row = layout.row()
        row.prop(latkml004, "latkml004_ModelType")

        row = layout.row()
        row.prop(latkml004, "latkml004_lineThreshold")

        row = layout.row()
        row.prop(latkml004, "latkml004_distThreshold")

        row = layout.row()
        row.prop(latkml004, "latkml004_csize")
        row.prop(latkml004, "latkml004_maxIter")

        row = layout.row()
        row.prop(latkml004, "latkml004_thickness")

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

def npToCv(img):
    img = img.astype(np.float32)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cvToNp(img):
    return np.asarray(img)

def cvToBlender(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    blender_image = bpy.data.images.new("Image", width=rgb_image.shape[1], height=rgb_image.shape[0])
    pixels = np.flip(rgb_image.flatten())
    blender_image.pixels.foreach_set(pixels)
    blender_image.update()
    return blender_image

def renderFrame(_format="PNG"):
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    output_path = os.path.join(bpy.app.tempdir, "render.png")
    bpy.context.scene.render.filepath = output_path

    oldFormat = bpy.context.scene.render.image_settings.file_format
    bpy.context.scene.render.image_settings.file_format = _format
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.image_settings.file_format = oldFormat

def renderToCv():
    renderFrame()
    image_path = bpy.context.scene.render.filepath
    image = cv2.imread(image_path)
    return image

def renderToNp():
    renderFrame()
    image_path = bpy.context.scene.render.filepath
    image = bpy.data.images.load(image_path)
    image_array = np.array(image.pixels[:])
    image_array = image_array.reshape((height, width, 4))
    image_array = np.flipud(image_array)
    image_array = image_array[:, :, :3]
    return image_array

def remap(value, min1, max1, min2, max2):
    '''
    range1 = max1 - min1
    range2 = max2 - min2
    valueScaled = float(value - min1) / float(range1)
    return min2 + (valueScaled * range2)
    '''
    return np.interp(value,[min1, max1],[min2, max2])

def loadModel():
    latkml004 = bpy.context.scene.latkml004_settings

    animeModel = "anime_style_512x512.onnx"
    contourModel = "contour_style_512x512.onnx"
    opensketchModel = "opensketch_style_512x512.onnx"
    experimentalModel = "latest_net_G.onnx"
    
    whichModel = animeModel

    if (latkml004.latkml004_ModelStyle.lower() == "contour"):
        whichModel = contourModel
    elif (latkml004.latkml004_ModelStyle.lower() == "opensketch"):
        whichModel = opensketchModel
    elif (latkml004.latkml004_ModelStyle.lower() == "experimental"):
        whichModel = experimentalModel

    if (latkml004.latkml004_ModelType.lower() == "pix2pix"):
        return Pix2pix(os.path.join(findAddonPath(), os.path.join("onnx", whichModel)))
    else:
        return Informative_Drawings(os.path.join(findAddonPath(), os.path.join("onnx", whichModel)))

# https://blender.stackexchange.com/questions/262742/python-bpy-2-8-render-directly-to-matrix-array
# https://blender.stackexchange.com/questions/2170/how-to-access-render-result-pixels-from-python-script/3054#3054
def doInference(onnx):
    latkml004 = bpy.context.scene.latkml004_settings

    img_cv = renderToCv()
    result = onnx.detect(img_cv)
    
    outputUrl = os.path.join(bpy.app.tempdir, "output.png")
    cv2.imwrite(outputUrl, result)

    '''
    lineThreshold = 64
    csize = 10
    '''
    maxIter = 999

    im0 = cv2.imread(outputUrl)
    im0 = cv2.bitwise_not(im0) # invert
    imWidth = len(im0[0])
    imHeight = len(im0)
    im = (im0[:,:,0] > latkml004.latkml004_lineThreshold).astype(np.uint8)
    im = skeletonize(im).astype(np.uint8)
    polys = from_numpy(im, latkml004.latkml004_csize, latkml004.latkml004_maxIter)

    laFrame = latk.LatkFrame(frame_number=bpy.context.scene.frame_current)

    scene = bpy.context.scene
    camera = bpy.context.scene.camera

    frame = camera.data.view_frame(scene=bpy.context.scene)
    topRight = frame[0]
    bottomRight = frame[1]
    bottomLeft = frame[2]
    topLeft = frame[3]

    resolutionX = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
    resolutionY = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))
    xRange = np.linspace(topLeft[0], topRight[0], resolutionX)
    yRange = np.linspace(topLeft[1], bottomLeft[1], resolutionY)

    originalStrokes = []
    originalStrokeColors = []
    separatedStrokes = []
    separatedStrokeColors = []

    for target in bpy.data.objects:
        if target.type == "MESH":
            matrixWorld = target.matrix_world
            matrixWorldInverted = matrixWorld.inverted()
            origin = matrixWorldInverted @ camera.matrix_world.translation

            for stroke in polys:
                newStroke = []
                newStrokeColor = []
                for point in stroke:
                    rgbPixel = img_cv[point[1]][point[0]]
                    rgbPixel2 = (rgbPixel[2], rgbPixel[1], rgbPixel[0], 1)

                    xPos = remap(point[0], 0, resolutionX, xRange.min(), xRange.max())
                    yPos = remap(point[1], 0, resolutionY, yRange.max(), yRange.min())
                   
                    pixelVector = Vector((xPos, yPos, topLeft[2]))
                    pixelVector.rotate(camera.matrix_world.to_quaternion())
                    destination = matrixWorldInverted @ (pixelVector + camera.matrix_world.translation) 
                    direction = (destination - origin).normalized()
                    hit, location, norm, face = target.ray_cast(origin, direction)

                    if hit:
                        location = target.matrix_world @ location
                        co = (location.x, location.y, location.z)
                        newStroke.append(co)
                        newStrokeColor.append(rgbPixel2)

                if (len(newStroke) > 1):
                    originalStrokes.append(newStroke)
                    originalStrokeColors.append(newStrokeColor)

        for i in range(0, len(originalStrokes)):
            separatedTempStrokes, separatedTempStrokeColors = separatePointsByDistance(originalStrokes[i], originalStrokeColors[i], latkml004.latkml004_distThreshold)

            for j in range(0, len(separatedTempStrokes)):
                separatedStrokes.append(separatedTempStrokes[j])
                separatedStrokeColors.append(separatedTempStrokeColors[j])

        for i in range(0, len(separatedStrokes)):
            laPoints = []
            for j in range(0, len(separatedStrokes[i])):
                laPoint = latk.LatkPoint(separatedStrokes[i][j])
                laPoint.vertex_color = separatedStrokeColors[i][j]
                laPoints.append(laPoint)

            if (len(laPoints) > 1):
                laFrame.strokes.append(latk.LatkStroke(laPoints))

    return laFrame

def separatePointsByDistance(points, colors, threshold):
    if (len(points) != len(colors)):
        return None

    separatedPoints = []
    separatedColors = []
    currentPoints = []
    currentColors = []

    for i in range(0, len(points) - 1):
        currentPoints.append(points[i])
        currentColors.append(colors[i])

        distance = lb.getDistance(points[i], points[i + 1])

        if (distance > threshold):
            separatedPoints.append(currentPoints)
            separatedColors.append(currentColors)
            currentPoints = []
            currentColors = []

    currentPoints.append(points[len(points) - 1])
    currentColors.append(colors[len(colors) - 1])
    separatedPoints.append(currentPoints)
    separatedColors.append(currentColors)

    return separatedPoints, separatedColors

def setThickness(thickness):
    gp = lb.getActiveGp()
    bpy.ops.object.gpencil_modifier_add(type="GP_THICK")
    gp.grease_pencil_modifiers["Thickness"].thickness_factor = thickness 
    bpy.ops.object.gpencil_modifier_apply(apply_as="DATA", modifier="Thickness")

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

class Informative_Drawings():
    def __init__(self, modelpath):
        try:
            cv_net = cv2.dnn.readNet(modelpath)
        except:
            print('opencv read onnx failed!!!')
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        #self.net = onnxruntime.InferenceSession(modelpath, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
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

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/1113
class Pix2pix():
    def __init__(self, onnx_file):
        self.net = onnxruntime.InferenceSession(onnx_file)
        self.input_size = 256
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name
        print("input_name = " + self.input_name)
        print("output_name = " + self.output_name)

    def detect(self, image):
        if isinstance(image, str):
            image=cv2.imdecode(np.fromfile(image, dtype=np.uint8), -1)
        elif isinstance(image, np.ndarray):
            image=image.copy()
        # image=image[0:256, 0:256]
        img = cv2.resize(image, (self.input_size, self.input_size))
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image / 255.0
        input_image = (input_image - 0.5) / 0.5 
        input_image = input_image.astype('float32')
        print(input_image.shape)
        # x = x[None,:,:,:]
        outs = self.net.run(None, {self.input_name: input_image})[0].squeeze(axis=0)
        outs = np.clip(((outs*0.5+0.5) * 255), 0, 255).astype(np.uint8) 
        outs = outs.transpose(1, 2, 0).astype('uint8')
        outs = cv2.cvtColor(outs, cv2.COLOR_RGB2BGR)
        outs=np.hstack((img, outs))
        print("outs",outs.shape)
        # return cv2.resize(outs, (image.shape[1], image.shape[0]))
        return outs

'''
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
'''

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