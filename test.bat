@echo off

set STYLE=anime_style
rem STYLE=opensketch_style
set RGB_PATH=input/rgb
set RESULT_PATH=results/%STYLE%
set MAX_FRAMES=999
set RENDER_RES=480

rmdir /s /q results\%STYLE%

python test.py --name %STYLE% --dataroot %RGB_PATH% --how_many %MAX_FRAMES% --size %RENDER_RES%

set DEPTH_PATH=input/depth
set LINE_THRESHOLD=64
set USE_SWIG=0

python skeletonizer.py -- %RESULT_PATH% %RGB_PATH% %DEPTH_PATH% %LINE_THRESHOLD% %USE_SWIG%



@pause