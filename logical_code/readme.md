# Aims
this is a logical code design for fix length convolution with mxnet ndarray. 

* we compress weight tensor to a fixed state which has a certain number of parameters in every filters.

* then, we get a mask that indicate the exact location of valid parameters in filters.

* after we prepare the data, we can use this code to inference. However, the code is rather slow in practice for it can not bypass general im2col function. For that, we prepare another re-adjust im2col which can be compiled with mxnet. The workflow how to compile a third party code in mxnet can follow [this](http://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU). Before compiling, the code write in c++ should copy to path *incubator-mxnet/src/operator/contrib/nn/*
