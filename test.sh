STYLE=anime_style
#STYLE=opensketch_style
RGB_PATH=input/rgb
RESULT_PATH=results/$STYLE
MAX_FRAMES=999
RENDER_RES=512 # 512

rm -rf $RESULT_PATH

python test.py --name $STYLE --dataroot $RGB_PATH --how_many $MAX_FRAMES --size $RENDER_RES

DEPTH_PATH=input/depth
LINE_THRESHOLD=64 # 64

python skeletonizer.py -- $RESULT_PATH $RGB_PATH $DEPTH_PATH $LINE_THRESHOLD

