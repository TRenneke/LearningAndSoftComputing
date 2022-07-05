from ctypes.wintypes import SIZE
import bpy
import random
import math
import numpy as np
import numpy.linalg as la
import os
import json
from pprint import pprint
deg2rad = math.pi / 180

#Settings:
out_dir = ""
sign_dir = ""
num_imgs = 1000


def randRange(v):
    return random.random() * (v[1] - v[0]) + v[0]

def randRotation(obj, ranges):
    range = (-30 * deg2rad , 30 * deg2rad)
    obj.rotation_euler = [randRange(ranges[0]) * deg2rad, randRange(ranges[1]) * deg2rad, randRange(ranges[2]) * deg2rad]

def randPosition(obj, ranges):
    for idx, range in enumerate(ranges): 
        obj.location[idx] = randRange(range)

def randLightColor(obj):
    #obj = bpy.data.objects["test"].
    obj.data.color = [randRange((0, 1)), randRange((0, 1)), randRange((0, 1))]

def randObjColors(objs):
    for o in objs:
        randLightColor(o)
        
def randPositions(objs, ranges):
    for o in objs:
        randPosition(o, ranges)
        
def randPosInCam(objs, ranges, cam):
    for obj in objs:
        cam_mat = cam.matrix_world.inverted()
        proj_mat = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get())
        mat = proj_mat @ cam_mat
        
        #proj_mat = np.reshape(np.array(list(proj_mat)))
        
        zpos = np.array([1, 0, randRange(ranges[2]), 1])
        zpos = np.reshape(np.array(list(proj_mat)), (4, 4)) @ zpos
        xpos = zpos[0] / zpos[3]
        zpos = zpos[2] / zpos[3]
        for i in range(2):
            ranges[i] = (ranges[i][0] + xpos, ranges[i][1] - xpos)
        pos = np.array([randRange(ranges[0]), randRange(ranges[1]), zpos, 1])
        pos = np.reshape(np.array(list(mat.inverted())), (4, 4)) @ pos
        pos = pos[0:3] / pos[3]
        for idx, p in enumerate(pos):
            obj.location[idx] = p
def setEnergy(objs, energy):
    for obj in objs:
        obj.data.energy = energy        
def randCameraPos(cam):
    cam.location = [0, 0, 5]
    cam.keyframe_insert(data_path="location", frame=1)
    randPosition(cam, [(-0.5, 0.5), (-0.5, 0.5), (1.5, 9.5)])
    cam.keyframe_insert(data_path="location", frame=2)
    cam.location = [0, 0, 5]
    
def randFocalLength(cam):
    r = randRange((10, 100))
    cam.data.lens = r
    return r
    
def renderImage(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    
def getObjects(name):
    r = []
    for k in bpy.data.objects.keys():
        if name in k:
            r.append(bpy.data.objects[k])
    return r
def setTrafficSign(file):
    bpy.data.images["TrafficSign"].filepath = file
lightRanges = [(-5, 5), (-5, 5), (-5, 5)]
def squareRandRange(range):
    min = range[0]
    max = range[1]
    return min + (max - min) * pow(random.random(), 2)
def randBackgroundTex(mat, textures):
    mat.node_tree.nodes["Vector Math.001"].inputs[1].default_value[0] = random.random()
    mat.node_tree.nodes["Vector Math.001"].inputs[1].default_value[1] = random.random()
    scale = random.random()
    mat.node_tree.nodes["Vector Math"].inputs[1].default_value[0] = scale
    mat.node_tree.nodes["Vector Math"].inputs[1].default_value[1] = scale
    
    mat.node_tree.nodes["Mix"].inputs[0].default_value = random.random() * 0.25
    
    dir = os.path.join("textures", textures[random.randrange(0, len(textures))])
    for f in os.listdir(dir):
        if "NormalGL" in f:
            bpy.data.images["BackgroundNormal"].filepath = os.path.join("//" + dir, f)
            print("changed normal")
        elif "Color" in f:
            bpy.data.images["BackgroundColor"].filepath = os.path.join("//" + dir, f)
            print("changed color")
        else:
            print(f)

    #mat.node_tree.nodes["Vector Math.001"].inputs[2].default_value[0] = random.random()
    #mat.node_tree.nodes["Vector Math.001"].inputs[3].default_value[0] = random.random()

def saveMatrix(obj: bpy.types.Object, camera: bpy.types.Object) -> np.ndarray:
    #scene = bpy.context.scene
    #camera = scene.camera
    cam_mat = camera.matrix_world.inverted()
    proj_mat = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get())
    obj_mat = obj.matrix_world
    mat = proj_mat @ cam_mat @ obj_mat
    r = []
    for v in list(mat):
        r.append(list(v))
    return r
#def generateSigns()
class Scene:
    def __init__(self) -> None:
        self.lights = getObjects("Light")
        self.cam = bpy.data.objects["Camera"]
        self.sign = bpy.data.objects["Sign"]
        self.mat = bpy.data.materials["SignMaterial"] 
        self.textures = os.listdir("textures")
    def randomize(self):
        randRotation(self.sign, [(-3, 3), (-30, 30), (-3, 3)])
        randPositions(self.lights, lightRanges)
        randObjColors(self.lights)
        setEnergy(self.lights, squareRandRange((5, 1000)))
        
        randCameraPos(self.cam)
        f = randFocalLength(self.cam)
        randPosInCam([self.sign], [(-1.0, 1.0), (-1.0, 1.0), (f*(-0.1), f*(-1.0))], self.cam)
        
        randBackgroundTex(self.mat, self.textures)
    def get_mat(self):
        return saveMatrix(self.sign, self.cam)

def genereateAllSignImages(signFolder, count, img_path, ann_path, scene: Scene):
    l = len(os.listdir(signFolder))
    for i, path in enumerate(os.listdir(signFolder)):
        print(os.path.join(signFolder, path))
        setTrafficSign("//" + os.path.join(signFolder, path))
        generateSignImages(count, img_path, ann_path, path.split("_")[0], path.split("_")[1].split(".")[0], scene)    
        print(f"Sign: {i} / {l}")
def generateSignImages(count, img_path, ann_path, name, variant, scene: Scene):
    for index in range(count):
        scene.randomize()
        with open(os.path.join(ann_path, f"{name}_{variant}_{index}.info.json"), "w+") as file:
            file.write(json.dumps(saveMatrix(scene.sign, scene.cam)))
        
        renderImage(os.path.join(img_path, f"{name}_{variant}_{index}.png"))
        print(f"Image: {index} / {count}")
def main():
    bpy.context.scene.frame_set(1)
    scene = Scene()
    #generateSignImages(10, os.path.join("out", "imgs"), os.path.join("out", "info"), "test", os.path.join("//Signs", "MUTCD_R1-2.svg.png"), sign, cam, lights)
    genereateAllSignImages("Signs", 250, os.path.join("//out", "imgs"), os.path.join("out", "info"), scene)
    #scene.randomize()
    #for path in os.listdir("Signs"):
    #    print(os.path.join("Signs", path))
    #    setTrafficSign("//" + os.path.join("Signs", path))
if __name__ == "__main__":
    main()