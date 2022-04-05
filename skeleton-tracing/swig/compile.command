swig -python trace_skeleton.i

# OS X
PYTHON_PATH=/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7
PYTHON_INCLUDE=$PYTHON_PATH/include/python3.7m
PYTHON_LIB=$PYTHON_PATH/lib/libpython3.7m.dylib
gcc -O3 -c trace_skeleton.c trace_skeleton_wrap.c -I$PYTHON_INCLUDE
gcc $(python3-config --ldflags) -dynamiclib *.o -o _trace_skeleton.so -I$PYTHON_LIB -undefined dynamic_lookup

# quick tests
# python3 -i -c "import trace_skeleton; trace_skeleton.trace('\0\0\0\1\1\1\0\0\0',3,3); print(trace_skeleton.len_polyline());"
# python3 -i -c "import trace_skeleton; print(trace_skeleton.from_list([0,0,0,1,1,1,0,0,0],3,3))"

python3 example.py