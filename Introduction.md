**JIT Algorithm Implementation and Analog In-Memory Computing Integration**

This implementation follows the JIT algorithm by Kaiming He (MIT) to generate images, but largely relies on analog in-memory computing, particularly for linear computations. The system typically uses a NOR-Flash array with INT8 weight precision.

For vector input in Matrix-Vector Multiplication (MVM), the vector is quantized into Int12 and then separated into the most significant bits (MSB-8) and least significant bits (LSB-8). Both parts undergo the same analog algorithmic process, with a 4-bit overlap to model noise-budget redundancy (physical bus still two 8-bit slices, effective precision ≈12 bits).

A 10-bit ADC is used to sample the algorithmic results (configurable up to 12-bit), and then perform shifting and operations in the digital domain.
