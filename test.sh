STYLE=anime_style
#STYLE=opensketch_style
RGB_PATH=input/rgb
MAX_REPS=999
RENDER_RES=512

rm -rf $RESULT_PATH

python test.py --name $STYLE --dataroot $RGB_PATH --how_many $MAX_REPS --size $RENDER_RES

RESULT_PATH=results/$STYLE
DEPTH_PATH=input/depth
LINE_THRESHOLD=64
LINE_QUALITY=10

python skeletonizer.py -- $RESULT_PATH $RGB_PATH $DEPTH_PATH $LINE_THRESHOLD $LINE_QUALITY

