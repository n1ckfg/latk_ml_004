STYLE=anime_style
#STYLE=opensketch_style

#python test.py --name $STYLE --dataroot input

RESULT_IMG=results/$STYLE/bridge_out.png
DEPTH_IMG=results/$STYLE/bridge_depth.png
RGB_IMG=results/$STYLE/bridge.png
THRESHOLD=64

python skeletonizer.py -- $RESULT_IMG $DEPTH_IMG $RGB_IMG $THRESHOLD

