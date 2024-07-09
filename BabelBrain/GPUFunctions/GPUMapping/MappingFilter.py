import logging
logger = logging.getLogger()
import os
import platform

import numpy as np

try:
    from GPUUtils import InitCUDA,InitOpenCL,InitMetal,get_step_size
except:
    from ..GPUUtils import InitCUDA,InitOpenCL,InitMetal,get_step_size

def InitMapFilter(DeviceName='A6000',GPUBackend='OpenCL'):
    global queue 
    global prgcl
    global sel_device
    global ctx
    global knl
    global mf
    global clp

    base_path = os.path.abspath('.')
    kernel_files = [
        base_path + os.sep + 'BabelBrain' + os.sep + 'GPUFunctions' + os.sep + 'GPUMapping' + os.sep + 'map_filter.cpp',
    ]

    if GPUBackend == 'CUDA':
        import cupy as cp
        clp = cp

        preamble = '#define _CUDA'
        ctx,prgcl,sel_device = InitCUDA(preamble,kernel_files,DeviceName)

        # Create kernels from program function
        knl = prgcl.get_function('mapfilter')

    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl
        clp = pocl

        preamble = '#define _OPENCL'
        queue,prgcl,sel_device,ctx,mf = InitOpenCL(preamble,kernel_files,DeviceName)
        
        # Create kernels from program function
        knl = prgcl.mapfilter

    elif GPUBackend == 'Metal':
        import metalcomputebabel as mc
        clp = mc

        preamble = '#define _METAL'
        prgcl, sel_device, ctx = InitMetal(preamble,kernel_files,DeviceName)

        # Create kernels from program function
        knl=prgcl.function('mapfilter')

    
def MapFilter(HUMap,SelBone,UniqueHU,GPUBackend='OpenCL'):

    logger.info(f"\nStarting Map Filter")
    
    assert(HUMap.dtype==np.float32)
    assert(UniqueHU.dtype==np.float32)
    assert(SelBone.dtype==np.uint8)
    assert(np.all(np.array(HUMap.shape)==np.array(SelBone.shape)))
    assert(np.isfortran(HUMap)==False) 
    output=np.zeros(HUMap.shape,np.uint32)
    int_params = np.zeros(5,dtype=np.uint32)
    int_params[0] = len(UniqueHU)
    int_params[1] = HUMap.shape[0]
    int_params[2] = HUMap.shape[1]
    int_params[3] = HUMap.shape[2]
 
    totalPoints = output.size
    step = get_step_size(sel_device,num_large_buffers=3,data_type=output.dtype,GPUBackend=GPUBackend)
    logger.info(f"Total points: {totalPoints}")
    
    for point in range(0,totalPoints,step):

        # Grab indices for current output section
        slice_start = (point // (output.shape[1] * output.shape[2]))
        slice_end = min(((point + step) // (output.shape[1] * output.shape[2])),output.shape[0])
        logger.info(f"Working on slices {slice_start} to {slice_end} out of {output.shape[0]}")
        
        # Grab sections of data
        HUMap_section = np.copy(HUMap[slice_start:slice_end,:,:])
        SelBone_section = np.copy(SelBone[slice_start:slice_end,:,:])
        output_section = np.copy(output[slice_start:slice_end,:,:])

        int_params[4] = output_section.size

        if GPUBackend=='CUDA':
            with ctx:

                HUMap_section_gpu = clp.asarray(HUMap_section)
                UniqueHU_gpu = clp.asarray(UniqueHU)
                SelBone_section_gpu = clp.asarray(SelBone_section)
                output_section_gpu = clp.asarray(output_section)
                int_params_gpu = clp.asarray(int_params)
                
                # Deploy kernel
                Block = (4,4,4)
                Grid = (HUMap_section.shape[0]//Block[0]+1,HUMap_section.shape[1]//Block[1]+1,HUMap_section.shape[2]//Block[2]+1)
                knl(Grid,Block,(HUMap_section_gpu,
                        SelBone_section_gpu,
                        UniqueHU_gpu,
                        output_section_gpu,
                        int_params_gpu))

                output_section=output_section_gpu.get()
        elif GPUBackend=='OpenCL':
            
            # Move input data from host to device memory
            HUMap_section_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=HUMap_section)
            UniqueHU_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=UniqueHU)
            SelBone_section_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SelBone_section)
            output_section_gpu = clp.Buffer(ctx, mf.WRITE_ONLY, output_section.nbytes)
            int_params_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int_params)

            # Deploy map filter kernel
            try:
                knl(queue, output_section.shape, 
                        None,
                        HUMap_section_gpu,
                        SelBone_section_gpu,
                        UniqueHU_gpu,
                        output_section_gpu,
                        int_params_gpu)
            except clp.MemoryError as e:
                raise MemoryError(f"{e}\nRan out of GPU memory, suggest lowering PPW")
            queue.finish()

            # Move kernel output data back to host memory
            clp.enqueue_copy(queue,output_section,output_section_gpu)
            queue.finish()

            # Release GPU memory
            HUMap_section_gpu.release()
            UniqueHU_gpu.release()
            SelBone_section_gpu.release()
            output_section_gpu.release()
            queue.finish()

        else:
            # prgcl = ctx.kernel("#define _METAL\n#define dimUnique %i\n" %(len(UniqueHU))+_code)
            # knl=prgcl.function('mapfilter')
            HUMap_pr = ctx.buffer(HUMap) 
            UniqueHU_pr = ctx.buffer(UniqueHU)
            SelBone_pr =  ctx.buffer(SelBone)
            CtMap_pr=  ctx.buffer(CtMap.nbytes)
            ctx.init_command_buffer()
            handle=knl(int(np.prod(HUMap.shape)),HUMap_pr,SelBone_pr,UniqueHU_pr,CtMap_pr )
            ctx.commit_command_buffer()
            ctx.wait_command_buffer()
            del handle
            if 'arm64' not in platform.platform():
                ctx.sync_buffers((CtMap_pr,UniqueHU_pr))
            CtMap=np.frombuffer(CtMap_pr,dtype=CtMap.dtype).reshape(CtMap.shape)
  
        # Record results in output array
        output[slice_start:slice_end,:,:] = output_section[:,:,:]

    return output

# if __name__ == "__main__":
#     from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
#     t=ReadFromH5py('test.h5')
#     InitCUDA('A6000')
#     MapFilter(t['HUMap'],t['SelBone'],t['UniqueHU'],GPUBackend='CUDA')