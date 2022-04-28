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
		# https://forum.open3d.org/t/azure-kinect-intrinsic-structure/121
		cx = 2044.2911376953125
		cy = 1565.8837890625
		fx = 1959.37890625
		fy = 1958.880126953125
	elif (name == "kinect_2"):
		# https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
		cx = 254.878
		cy = 205.395
		fx = 365.456
		fy = 365.456
	elif (name == "realsense_D435"):
		if (mode == "640x360"):
			# https://answers.ros.org/question/363236/intel-realsense-d435-intrinsic-parameters/
			cx = 320.818268
			cy = 178.779297
			fx = 322.282410
			fy = 322.282410
		elif (mode == "640x480"):
			# https://support.intelrealsense.com/hc/en-us/community/posts/4403651641491-about-ros-camera-calibration-and-d435-on-chip-calibration-values
			cx = 320.90576171875
			cy = 235.999221801758
			fx = 617.034790039063
			fy = 617.2119140625
		elif (mode == "848x480"):
			# https://github.com/IntelRealSense/realsense-ros/issues/1661
			cx = 423.014007568359
			cy = 239.275390625
			fx = 426.795440673828
			fy = 426.795440673828
		elif (mode == "1280x720"):
			# https://github.com/IntelRealSense/realsense-ros/issues/1661
			cx = 638.51171875
			cy = 358.90625
			fx = 644.219543457031
			fy = 644.219543457031

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