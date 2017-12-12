# Matrix multiplication using CUDA

This is a CUDA program to multiply two matrices P<sub>a x b</sub> and Q<sub>b x c</sub> to calculate R<sub>a x c</sub>. The values for a, b and c are specified by user and their default value is <code>1024</code>. The program also accepts the number of blocks and threads per blocks from user.

I have used MS Windows as my development environment. If you are using Linux, please check steps to build.

### System requirements
MS Windows environment:
1. MS Visual Studio
2. NVCC and Nsight

Linux:
1. NVCC

### How to build
MS Windows environment:
1. Import the kernel.cu file in MS Visual Studio
2. Set the value of <code>WINDOWS</code> to <code>1</code>. It has been <code>#define</code>d on first line of kernel.cu
3. Build the project

Linux:
1. Open kernel.cu and unset the value of <code>WINDOWS</code> to <code>0</code>. It has been <code>#define</code>d on first line of kernel.cu
2. Make the project
<pre>
    <code>
        $ cd matrix_mult
        $ make
    </code>
</pre>

### How to run
MS Windows environment:
1. Execute the project in MS Visual Studio using Nsight

Linux:
1. Execute kernel binary file
<pre>
    <code>
        $ cd matrix_mult
        $ ./kernel
    </code>
</pre>

<b>Note</b>: It accepts the values of a, b and c using command-line arguments if all three values are provided. Otherwise, the program uses default value.