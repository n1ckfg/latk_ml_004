bl_info = {
    "name": "latk-ml-004", 
    "author": "Nick Fox-Gieg",
	"version": (0, 0, 1),
	"blender": (3, 0, 0),
    "description": "Generate brushstrokes from a mesh using informative-drawings",
    "category": "Animation"
}

import bpy
from bpy.types import Operator, AddonPreferences
from bpy.props import (BoolProperty, FloatProperty, StringProperty, IntProperty, PointerProperty, EnumProperty)
from bpy_extras.io_utils import (ImportHelper, ExportHelper)

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
    bl_idname = "object.steve" #+ __name__
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
        pass
        return {'FINISHED'}

# https://blender.stackexchange.com/questions/167862/how-to-create-a-button-on-the-n-panel
class latkml004Properties_Panel(bpy.types.Panel):
    """Creates a Panel in the 3D View context"""
    bl_idname = "GREASE_PENCIL_PT_latkml004PropertiesPanel"
    bl_space_type = 'VIEW_3D'
    bl_label = "latk-ml-004"
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
        row.operator("latkml004_button.allframes")
        row.operator("latkml004_button.singleframe")
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

