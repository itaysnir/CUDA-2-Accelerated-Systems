#include "ex2.h"
#include <cuda/atomic>

__device__ void prefix_sum(int arr[], int arr_size) {
    const int tid = threadIdx.x; 
    int increment;

    for (int stride = 1 ; stride < arr_size ; stride *= 2)
    {
        if (tid >= stride && tid < arr_size)
        {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size)
        {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__
void get_histogram(int* hist, uchar* all_in, int tile_row, int tile_col)
{
    const int tid = threadIdx.x;
    const int thread_work = (TILE_WIDTH * TILE_WIDTH) / blockDim.x;
    const int threads_per_row = TILE_WIDTH / thread_work;
    const int x_index = (TILE_WIDTH * tile_row) + (tid / threads_per_row);
    const int y_index = (TILE_WIDTH * tile_col) + ((tid % threads_per_row) * thread_work);
    int color_value = 0;
    int index = 0;
    
    for(int j = 0 ; j < thread_work ; j++)
    {
        index = x_index * IMG_WIDTH + y_index + j;
        color_value = all_in[index];
        atomicAdd(&hist[color_value], 1);
    }  
}

__device__
void get_maps(int* cdf, uchar* maps, int tile_row, int tile_col)
{
    const int tid = threadIdx.x;
    if (tid >= COLOR_COUNT)
    {
        return;
    }

    const int tile_size = TILE_WIDTH * TILE_WIDTH;
    const int maps_start_index = ((tile_row * TILE_COUNT) + tile_col) * COLOR_COUNT;

    maps[maps_start_index + tid] = (float(cdf[tid]) * (COLOR_COUNT - 1)) / (tile_size);
}

__device__
void process_image(uchar *in, uchar *out, uchar* maps) 
{
    __shared__ int hist[COLOR_COUNT];

    for (int tile_row = 0 ; tile_row < TILE_COUNT ; tile_row++)
    {
        for (int tile_col = 0 ; tile_col < TILE_COUNT ; tile_col++)
        {
            memset(hist, 0, COLOR_COUNT * sizeof(int));
            __syncthreads();

            get_histogram(hist, in, tile_row, tile_col);
            __syncthreads();          
    
            prefix_sum(hist, COLOR_COUNT); 
            __syncthreads();            
        
            get_maps(hist, maps, tile_row, tile_col); 
            __syncthreads();    
        }
    }
    
    __syncthreads();
    interpolate_device(maps, in, out);

    return; 
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

/*
************************************************************************************* PART 1 - STREAMS SERVER *************************************************************************************
*/

class streams_server : public image_processing_server
{
private:
    static const int available_stream = -1;
    cudaStream_t streams[STREAM_COUNT];
    int stream_to_image[STREAM_COUNT];
    uchar* stream_to_map[STREAM_COUNT];
    uchar* stream_to_imgin[STREAM_COUNT];
    uchar* stream_to_imgout[STREAM_COUNT];

public:
    streams_server()
    {
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            stream_to_image[i] = available_stream;
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&stream_to_map[i], TILE_COUNT * TILE_COUNT * COLOR_COUNT));
            CUDA_CHECK(cudaMalloc(&stream_to_imgin[i], IMG_WIDTH * IMG_HEIGHT));
            CUDA_CHECK(cudaMalloc(&stream_to_imgout[i], IMG_WIDTH * IMG_HEIGHT));
        }
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            CUDA_CHECK(cudaFree(stream_to_map[i]));
            CUDA_CHECK(cudaFree(stream_to_imgin[i]));
            CUDA_CHECK(cudaFree(stream_to_imgout[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            if (stream_to_image[i] == available_stream)
            {
                stream_to_image[i] = img_id;
                CUDA_CHECK(cudaMemcpyAsync(stream_to_imgin[i], img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, streams[i]));
                process_image_kernel<<<1, THREADS_COUNT, 0, streams[i]>>>(img_in, stream_to_imgout[i], stream_to_map[i]);
                CUDA_CHECK(cudaMemcpyAsync(img_out, stream_to_imgout[i], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, streams[i]));
                return true;
            }
        }

        return false;
    }

    bool dequeue(int *img_id) override
    {
        for (int i = 0 ; i < STREAM_COUNT ; i++)
        {
            if (stream_to_image[i] != available_stream)
            {
            cudaError_t status = cudaStreamQuery(streams[i]); 
            switch (status) {
            case cudaSuccess:
                *img_id = stream_to_image[i];
                stream_to_image[i] = available_stream;
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

/*
************************************************************************************* PART 2 - QUEUES SERVER *************************************************************************************
*/

enum ImageType : int32_t
{
    BAD = -1,
    STOPPED = -2
};


struct ImageRequest
{
    int img_id;
    uchar *img_in;
    uchar *img_out;
};


class TASLock 
{
private:
    cuda::atomic<int, cuda::thread_scope_device> _lock;

public:
    __device__ TASLock() :
        _lock(0)
    {}

    __device__ void lock() 
    {
        while (_lock.exchange(1, cuda::memory_order_relaxed))
        {
            ;
        }
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
    }
 
    __device__ void unlock() 
    {
        _lock.store(0, cuda::memory_order_release);
    }
};


__device__ TASLock* gpu_lock;


template <typename T> 
class RingBuffer 
{
private:
    uint32_t N;
    cuda::atomic<size_t> _head;
    cuda::atomic<size_t> _tail;
    T* _mailbox;

public:
    RingBuffer() = default;
    explicit RingBuffer(int n):
        N(n),
        _head(0),
        _tail(0),
        _mailbox(nullptr)
    {  
        CUDA_CHECK(cudaMallocHost(&_mailbox, sizeof(T) * n));
    }
    ~RingBuffer()
    {
        CUDA_CHECK(cudaFreeHost(_mailbox));
    }

// In order to support non blocking enqueue, push must be non blocking too
    __device__ __host__ bool push(const T &data) 
    {
        int tail = _tail.load(cuda::memory_order_relaxed);
        if ((tail - _head.load(cuda::memory_order_acquire)) % (2 * N) == N)
        {
            return false;
        }

        _mailbox[_tail % N] = data;
        _tail.store(tail + 1, cuda::memory_order_release);
   
        return true;
}

    __device__ __host__ T pop() 
    {
        int head = _head.load(cuda::memory_order_relaxed);
        T item;

        if((_tail.load(cuda::memory_order_acquire) - head) % (2 * N) == 0)
        {
            item.img_id = ImageType::BAD;
            return item;
        }
    
        item = _mailbox[_head % N];
        _head.store(head + 1, cuda::memory_order_release);

        return item;
    }
};


__global__ void alloc_gpu_lock()
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    // allocate and initialize the gpu lock
    if (tid == 0 && bid == 0)
    {
        gpu_lock = new TASLock();
    }
}

__global__ void free_gpu_lock()
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    // free the gpu lock
    if (tid == 0 && bid == 0)
    {
        delete gpu_lock;
    }
}

// TODO implement a lock - Done
// TODO implement a MPMC queue - Done
// TODO implement the persistent kernel - Done
// TODO implement a function for calculating the threadblocks count - Done

__global__ void persistent_gpu_kernel(RingBuffer<ImageRequest>* cpu_to_gpu_queue, RingBuffer<ImageRequest>* gpu_to_cpu_queue, uchar* maps)
{
    // TODO - implement a kernel that simply listens to the cpu_to_gpu queue, pops any pending requests, and executes it on a single TB. 
    // After its completion, writes the result back to the gpu_to_cpu queue
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    uchar* tb_map = maps + bid * MAP_SIZE; 
    __shared__ ImageRequest req;

    while(true)
    {
        if (tid == 0)
        {
            gpu_lock->lock();
            req = cpu_to_gpu_queue->pop(); 
            gpu_lock->unlock();
        }
        __syncthreads();

        // Halt all threads within this thread block
        if (req.img_id == ImageType::STOPPED)
        {
            return; 
        }

        if (req.img_id != ImageType::BAD)
        {
            process_image(req.img_in, req.img_out, tb_map);
            __syncthreads();
            if (tid == 0)
            {
                gpu_lock->lock();
                while(gpu_to_cpu_queue->push(req) == false)
                {
                    ;
                }
                gpu_lock->unlock();
            }
        }
    }
}

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    RingBuffer<ImageRequest>* cpu_to_gpu_queue;
    RingBuffer<ImageRequest>* gpu_to_cpu_queue;
    uchar* maps;
    uint32_t blocks_count;
    double queue_size;


private:
    static uint32_t get_threadblock_count(int threads_per_block)
    {
        /*
        For our GPU setup (sm-75):
        The GPU supports up to 65536 threads on all thread blocks. 
        Moreover, the amount of possible shared mem is 64KB. Since each TB uses 2KB shared mem, this constraint leaders to 32 TBs. 
        The last constraint is the regs count - total of 65536 regs are available for each SM. 
        According to nvcc nvlink-options: our implementation uses 29 regs (we can assume it rounds up to 32, due makefile), 40 stack, 160B gmem, 2KB smem, 376B cmem, 0B lmem
        According to nvcc ptxas-options: 29 regs, 32B stack frame, 28B spill stores & loads. 
        
        Therefore, the limit for our setup is 65536 registers / 32 (threads count) / threads_per_block
        */
        cudaDeviceProp prop; 
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        struct cudaFuncAttributes func;
		CUDA_CHECK(cudaFuncGetAttributes(&func,process_image_kernel));

        const uint32_t regs_per_thread = 32;                          // upper bound of func.numRegs, as stated by makefile
        const uint32_t used_smem_per_block = func.sharedSizeBytes;    // 2KB - 1KB of our shared mem, 1KB of the given interpolate_device function
        //const uint32_t regs_per_block = prop.regsPerBlock;          // this value must be above threads_per_block (max 1024) * regs_per_thread (32), it is 65536
        //const uint32_t smem_per_block = prop.sharedMemPerBlock;     // this value must be above used_smem_per_block, it is 49152 
        const uint32_t sm_count = prop.multiProcessorCount;
        const uint32_t threads_per_sm = prop.maxThreadsPerMultiProcessor;
        const uint32_t smem_per_sm = prop.sharedMemPerMultiprocessor;
        const uint32_t regs_per_sm = prop.regsPerMultiprocessor;

        const uint32_t max_threads_count = min(regs_per_sm / regs_per_thread, threads_per_sm);       // hides the thread count constraint
        const uint32_t thread_block_count_regs = max_threads_count / threads_per_block; 
        const uint32_t thread_block_count_smem = smem_per_sm / used_smem_per_block;  
        
        return min(thread_block_count_regs, thread_block_count_smem) * sm_count;
    }

    static double get_queue_size(uint32_t tb_count)
    {
        const double num_1 = std::log(QUEUE_SIZE_FACTOR * tb_count); 
        const double num_2 = std::ceil(num_1 / std::log(2));
        const double queue_size = std::pow(2, num_2);

        return queue_size;   
    }

public:
    queue_server(int threads)
    {
        // TODO initialize host state
        blocks_count = get_threadblock_count(threads);    
        queue_size = get_queue_size(blocks_count);

        CUDA_CHECK(cudaMalloc(&maps, blocks_count * MAP_SIZE));
        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_queue, sizeof(RingBuffer<ImageRequest>)));
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_queue, sizeof(RingBuffer<ImageRequest>)));
        new(cpu_to_gpu_queue) RingBuffer<ImageRequest>(queue_size);
        new(gpu_to_cpu_queue) RingBuffer<ImageRequest>(queue_size);
        // create gpu lock
        alloc_gpu_lock<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        persistent_gpu_kernel<<<blocks_count, threads>>>(cpu_to_gpu_queue, gpu_to_cpu_queue, maps);
    }

    ~queue_server() override
    {
        // Kill all working blocks. Factor 2 for "vidue hariga"
        for (uint32_t i = 0 ; i < blocks_count * 2 ; i++)
        {
            enqueue(ImageType::STOPPED, nullptr, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize()); 
        // TODO free resources allocated in constructor
        cpu_to_gpu_queue->~RingBuffer<ImageRequest>();
        gpu_to_cpu_queue->~RingBuffer<ImageRequest>();
        CUDA_CHECK(cudaFree(maps));
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_queue));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_queue));
        // free gpu lock
        free_gpu_lock<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        ImageRequest req;
        req.img_id = img_id;
        req.img_in = img_in;
        req.img_out = img_out;  // Already allocated as shared host memory by the caller
        
        // must implement non blocking push for this to be non blocking
        return cpu_to_gpu_queue->push(req);
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        // must implement non blocking pop to support this
        ImageRequest req = gpu_to_cpu_queue->pop();
        if (req.img_id == ImageType::BAD)
        {
            return false;
        }

        // TODO return the img_id of the request that was completed.
        *img_id = req.img_id;

        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
