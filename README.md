# CUDA-2-Accelerated-Systems
This project's main goal is to learn advanced topics regarding CUDA. 

It simulates many clients, each creates a request for heavy computation on a remote server. 
The requests can be runned parallel on a GPU. 
I've demonstrated two main approaches for parallelism: 
1. CUDA streams
2. CUDA queues + acquire-release memory model
