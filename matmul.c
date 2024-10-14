#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"

#if !defined(N) || N % 16 != 0
#error "N must be #defined as a multiple of 16"
#endif

const char *matmul =	"__kernel void matmul(__global float *restrict z,	\n"
			"		      __global const float *restrict x,	\n"
			"		      __global const float *restrict y,	\n"
			"		      const int n)			\n"
			"{							\n"
			"	int i = get_global_id(1), j = get_global_id(0);	\n"
			"	float t = 0;					\n"
			"	for (int k = 0; k < n; k++)			\n"
			"		t += x[i * n + k] * y[k * n + j];	\n"
			"	z[i * n + j] = t;				\n"
			"}							\n";

int
main()
{
	const static size_t grange[] = { N, N }, lrange[] = { 16, 16 };
	static float x[N][N], y[N][N], z[N][N];
	const static int n = N;
	cl_platform_id platid;
	clGetPlatformIDs(1, &platid, NULL);
	cl_device_id devid;
	clGetDeviceIDs(platid, CL_DEVICE_TYPE_GPU, 1, &devid, NULL);
	cl_context ctx = clCreateContext(0, 1, &devid, NULL, NULL, NULL);
	cl_command_queue cmdq = clCreateCommandQueue(ctx, devid, 0, NULL);
	cl_program prog = clCreateProgramWithSource(ctx, 1, &matmul, NULL, NULL);
	clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	cl_kernel kern = clCreateKernel(prog, "matmul", NULL);
	cl_mem xd = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof x, x, NULL);
	cl_mem yd = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof y, y, NULL);
	cl_mem zd = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof (float[N][N]), NULL, NULL);
	clSetKernelArg(kern, 0, sizeof zd, (void *) &zd);
	clSetKernelArg(kern, 1, sizeof xd, (void *) &xd);
	clSetKernelArg(kern, 2, sizeof yd, (void *) &yd);
	clSetKernelArg(kern, 3, sizeof (int), (void *) &n);
	clEnqueueNDRangeKernel(cmdq, kern, 2, NULL, grange, lrange, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdq, zd, CL_TRUE, 0, sizeof z, z, 0, NULL, NULL); 
	clReleaseMemObject(xd);
	clReleaseMemObject(yd);
	clReleaseMemObject(zd);
	clReleaseKernel(kern);
	clReleaseProgram(prog);
	clReleaseCommandQueue(cmdq);
	clReleaseContext(ctx);
	exit(EXIT_SUCCESS);
}
