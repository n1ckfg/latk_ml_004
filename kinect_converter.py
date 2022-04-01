import numpy as np

def getIntrinsics(name="generic", mode="default"):
	name = name.lower()
	mode = mode.lower()

	# Generic
	# https://en.wikipedia.org/wiki/Camera_resectioning
	# https://docs.opencv.org/4.x/dc/d3a/classcv_1_1viz_1_1Camera.html#a753809aff611cdcc5a8a377126156b05
	cx = 320 # principal point x
	cy = 240 # principal point y
	fx = 525 # focal length x
	fy = 525 # focal length y

	if (name == "azure_kinect"):
		# Azure Kinect 4K
		# https://forum.open3d.org/t/azure-kinect-intrinsic-structure/121
		cx = 2044.2911376953125
		cy = 1565.8837890625
		fx = 1959.37890625
		fy = 1958.880126953125
	elif (name == "kinect_2"):
		# Kinect 2
		# https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
		cx = 254.878
		cy = 205.395
		fx = 365.456
		fy = 365.456

	return cx, cy, fx, fy

def uvd_to_xyz(u, v, d, scale=1, name="generic", mode="default"):
    cx, cy, fx, fy = getIntrinsics(name, mode)

    d *= scale
    x_over_z = (cx - u) / fx
    y_over_z = (cy - v) / fy

    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    #print((x, y, z))

    return (x, y, z)