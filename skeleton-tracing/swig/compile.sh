PYTHON_VERSION=3.8

# UBUNTU
PYTHON_INCLUDE=/usr/include/python$PYTHON_VERSION
PYTHON_LIB=/usr/lib/x86_64-linux-gnu/libpython$PYTHON_VERSION.so
PYTHON_FLAGS="-undefined dynamic_lookup"

# OS X
#PYTHON_PATH=/usr/local/Cellar/python/$PYTHON_VERSION_1/Frameworks/Python.framework/Versions/$PYTHON_VERSION
#PYTHON_INCLUDE=$PYTHON_PATH/include/python$PYTHON_VERSIONm
#PYTHON_LIB=$PYTHON_PATH/lib/libpython$PYTHON_VERSIONm.dylib
#PYTHON_FLAGS="-undefined dynamic_lookup"

swig -python trace_skeleton.i
gcc -O3 -c trace_skeleton.c trace_skeleton_wrap.c -I$PYTHON_INCLUDE
gcc $(python3-config --ldflags) -dynamiclib *.o -o _trace_skeleton.so -I$PYTHON_LIB $PYTHON_FLAGS

# quick tests
# python3 -i -c "import trace_skeleton; trace_skeleton.trace('\0\0\0\1\1\1\0\0\0',3,3); print(trace_skeleton.len_polyline());"
# python3 -i -c "import trace_skeleton; print(trace_skeleton.from_list([0,0,0,1,1,1,0,0,0],3,3))"

#python3 example.py