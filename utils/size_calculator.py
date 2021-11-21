# kernel_size = 3
# stride = 2
# padding = 1
# input_size = 360
# dilation = 1

# def Calc_conv2d_size(kernel_size, stride, padding, input_size, dilation):
#   return(((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride) + 1)

# Calc_conv2d_size(kernel_size, stride, padding, input_size, dilation)

kernel_size = 3
stride = 1
padding = 1
dilation = 1
input_size = 180

def Calc_trans2d_size(kernel_size, stride, padding, input_size, dilation):
  return((input_size - 1)*stride - 2*padding + dilation*(kernel_size - 1) + 1)

print(Calc_trans2d_size(kernel_size, stride, padding, input_size, dilation))