**JIT Algorithm Implementation and Analog In-Memory Computing Integration**

This implementation follows the JIT algorithm by Kaiming He (MIT) to generate images, but largely relies on analog in-memory computing, particularly for linear computations. 

The system typically uses a NOR-Flash array with INT8 weight precision.For vector input in Matrix-Vector Multiplication (MVM), the vector is quantized into INT12 and then separated into 7-bit signed MSB and 5-bit unsigned LSB slices. The MSB pass can be sampled multiple times (default 2) to average out thermal noise.
