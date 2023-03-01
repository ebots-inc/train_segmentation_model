export mrtf2d=/home/maskrcnn_tf2/src
export MRTF2_SAVE_PATH=/home
# export PYTHONPATH=$mrtf2d/preprocess:$mrtf2d/common
export PYTHONPATH=$mrtf2d
export LD_LIBRARY_PATH=/TensorRT-8.4.3.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
export LD_LIBRARY_PATH=/TensorRT-8.4.3.1/targets/x86_64-linux-gnu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 