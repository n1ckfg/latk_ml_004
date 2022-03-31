STYLE=anime_style
#STYLE=opensketch_style

RGB_PATH=input/rgb
DEPTH_PATH=input/depth
RESULT_PATH=results/$STYLE
THRESHOLD=64

rm -rf $RESULT_PATH

python test.py --name $STYLE --dataroot $RGB_PATH --how_many 999

python skeletonizer.py -- $RESULT_PATH $RGB_PATH $DEPTH_PATH $THRESHOLD

