
# semi global matching on gpu (pytorch build)
original repo: https://github.com/dhernandez0/sgm<br/>
Result does not match the original perfectly<br/>

# setup
python setup.py build develop<br/>

# Identified issues
If your version of pytorch is higher, comment out #include <THC/THC.h> in line 12 of semi_global_matching_cuda.cu file and insert #include <ATen/cuda/CUDAEvent.h> below it.
