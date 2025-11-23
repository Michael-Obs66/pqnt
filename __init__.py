from pquant import (
    quantize_array,
    quantize_tensor,
    quantize_tensor_per_channel
)

# Import wrappers framework
from .pytorch import quantize_pytorch
from .onnx import quantize_onnx
from .tensorflow import quantize_tensorflow
