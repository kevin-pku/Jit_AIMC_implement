**JIT Algorithm Implementation and Analog In-Memory Computing Integration**

This implementation follows the JIT algorithm by Kaiming He (MIT) to generate images, but largely relies on analog in-memory computing, particularly for linear computations. 

The system typically uses a NOR-Flash array with INT8 weight precision and **Hybrid Sparse INT10 Activaction** precision.For vector input in Matrix-Vector Multiplication (MVM), the vector is quantized and then separated into 2-bit signed MSB (digital and sparse) and 8-bit signed LSB (analog) slices. The LSB pass can be sampled multiple times (default 2) to average out thermal noise.
