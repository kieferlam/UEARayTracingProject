#include "CLKernel.h"
#include <iomanip>

void CLKernel::display_kernel_info() {
	// Get kernel info
	size_t workgroup_size, pref_workgroup_multiple;
	cl_ulong private_mem_size, local_mem_size;
	clGetKernelWorkGroupInfo(getKernel(), cl::device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, NULL);
	clGetKernelWorkGroupInfo(getKernel(), cl::device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &pref_workgroup_multiple, NULL);
	clGetKernelWorkGroupInfo(getKernel(), cl::device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	clGetKernelWorkGroupInfo(getKernel(), cl::device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &private_mem_size, NULL);

	std::cout << "Kernel " << getKernelName() << ":" << std::endl;
	std::cout << std::setw(48) << "CL_KERNEL_WORK_GROUP_SIZE" << std::setw(8) << workgroup_size << std::endl;
	std::cout << std::setw(48) << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" << std::setw(8) << pref_workgroup_multiple << std::endl;
	std::cout << std::setw(48) << "CL_KERNEL_LOCAL_MEM_SIZE" << std::setw(8) << local_mem_size << std::endl;
	std::cout << std::setw(48) << "CL_KERNEL_PRIVATE_MEM_SIZE" << std::setw(8) << private_mem_size << std::endl;

}
