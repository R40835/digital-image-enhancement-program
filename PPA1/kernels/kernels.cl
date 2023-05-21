//histogram implementation
kernel void histogram(global const uchar* A, global int* H) {
    int id = get_global_id(0);

    //assumes that H has been initialised to 0
    int bin_index = A[id];//take value as a bin index

    atomic_add(&H[bin_index], 1);
    //Atomic operations deal with race conditions, but serialise the access to global memory, and are slow
}

//cumulative Histogram
//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void cum_hist(global int* H, global int* CH) {
    int id = get_global_id(0);
    int N = get_global_size(0);

    int sum = 0;
    for (int i = 0; i <= id; i++)
        sum += H[i];

    CH[id] = sum;
}

//LUT
kernel void LUT(global int* CH, global int* LUT) {
    int id = get_global_id(0);
    LUT[id] = (int)((double)CH[id] * 255.0 / (double)CH[255]);
}

//changing pixel values
//copies all pixels from A to B
kernel void re_project(global uchar* A, global int* LUT, global uchar* B) {
    int id = get_global_id(0);
    B[id] = (uchar)LUT[A[id]];
}
