@echo off

rem https://www.swig.org/Doc1.3/Windows.html

rem provide paths to
rem https://github.com/skeeto/w64devkit
rem https://www.swig.org/download.html

swig -python trace_skeleton.i

rem WINDOWS
rem https://github.com/LingDong-/skeleton-tracing/issues/14
rem gcc -fPIC -O3 -c trace_skeleton.c trace_skeleton_wrap.c -I/$(python3-config --cflags)
rem g++ -shared $(python3-config --cflags --ldflags) *.o -o _trace_skeleton.so
gcc -fPIC -O3 -c trace_skeleton.c trace_skeleton_wrap.c -I/C:\Users\nick\miniconda3\include
g++ -shared *.o -o _trace_skeleton.pyd

rem OS X
rem PYTHON_PATH=/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7
rem PYTHON_INCLUDE=$PYTHON_PATH/include/python3.7m
rem PYTHON_LIB=$PYTHON_PATH/lib/libpython3.7m.dylib
rem gcc -O3 -c trace_skeleton.c trace_skeleton_wrap.c -I$PYTHON_INCLUDE
rem gcc $(python3-config --ldflags) -dynamiclib *.o -o _trace_skeleton.so -I$PYTHON_LIB -undefined dynamic_lookup

rem quick tests
rem python3 -i -c "import trace_skeleton; trace_skeleton.trace('\0\0\0\1\1\1\0\0\0',3,3); print(trace_skeleton.len_polyline());"
rem python3 -i -c "import trace_skeleton; print(trace_skeleton.from_list([0,0,0,1,1,1,0,0,0],3,3))"

python3 example.py

@pause
