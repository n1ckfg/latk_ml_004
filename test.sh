STYLE=anime_style
#STYLE=opensketch_style

#python test.py --name $STYLE --dataroot examples/test

python skeletonizer.py -- results/$STYLE/bridge_out.png 64

