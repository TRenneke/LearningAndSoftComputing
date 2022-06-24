from ctypes.wintypes import SIZE
import bpy
import random
import math
import numpy as np
import numpy.linalg as la
import os
import json
deg2rad = math.pi / 180

#Settings:
out_dir = ""
sign_dir = ""
num_imgs = 1000


def randRange(v):
    return random.random() * (v[1] - v[0]) + v[0]
def randRotation(obj):
    range = (-15 * deg2rad , 15 * deg2rad)
    obj.rotation_euler = [randRange(range), randRange(range), randRange(range)]
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

def randCameraPos(cam):
    cam.location = [0, 0, 5]
    cam.keyframe_insert(data_path="location", frame=1)
    randPosition(cam, [(-0.5, 0.5), (-0.5, 0.5), (1.5, 9.5)])
    cam.keyframe_insert(data_path="location", frame=2)
    cam.location = [0, 0, 5]
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
    def randomize(self):
        randRotation(self.sign)
        randPositions(self.lights, lightRanges)
        randObjColors(self.lights)
        randCameraPos(self.cam)
    def get_mat(self):
        return saveMatrix(self.sign, self.cam)

def genereateAllSignImages(signFolder, count, img_path, ann_path, scene: Scene):
    for path in os.listdir(signFolder):
        print(os.path.join(signFolder, path))
        setTrafficSign("//" + os.path.join(signFolder, path))
        generateSignImages(count, img_path, ann_path, path.split("_")[0], scene)    
def generateSignImages(count, img_path, ann_path, name, scene: Scene):
    for index in range(count):
        scene.randomize()
        with open(os.path.join(ann_path, f"{name}_{index}.info.json"), "w+") as file:
            file.write(json.dumps(saveMatrix(scene.sign, scene.cam)))
        
        renderImage(os.path.join(img_path, f"{name}_{index}.png"))
def main():
    scene = Scene()
    #generateSignImages(10, os.path.join("out", "imgs"), os.path.join("out", "info"), "test", os.path.join("//Signs", "MUTCD_R1-2.svg.png"), sign, cam, lights)
    genereateAllSignImages("Signs", 200, os.path.join("//out", "imgs"), os.path.join("out", "info"), scene)
if __name__ == "__main__":
    main()