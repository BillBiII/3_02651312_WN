# 3_02651312_WN
This repository provides References, Derivations and Python code related to my Individual Research Project (M1R).

For the Citations_for_Poster.pdf, please download the PDF in order to access URLs.

For quantumlogicgates.py, I would advise you run this on a Linux system, or a simulated Linux environment such as Ubuntu, utilising WSL:Ubuntu (WSL meaning Windows Subsystem for Linux). This is due to the instability of NVIDIA's CUDA-Q v13.2.
This code was used to obtain the results for my poster, and the code may be incomplete or missing proper explanations, as it is not the main focus of the project.
GIF generation lines are commented out to vastly reduce run time of code.
Note that if you run the Python file through a Python Debugger, say in VSCode, it may fail when it is ran for the first time, due to line 23. If you find a solution to this, thanks.

The 3_02651312_WN.pdf file has been added as a backup file of the M1R poster.

derivations_WIP.pdf is missing its final chapter. This will be updated in the following days.

The following information may be useful for those wanting to run the quantumlogicgates.py file:
Package                  Version
------------------------ -----------
astpretty                3.0.0
certifi                  2026.5.20
charset-normalizer       3.4.7
contourpy                1.3.3
cuda-quantum-cu12        0.14.2
cudensitymat-cu12        0.5.2
cupy-cuda12x             13.6.0
custatevec-cu12          1.13.1
cutensor-cu12            2.6.0
cutensornet-cu12         2.12.2
cycler                   0.12.1
dill                     0.4.1
fastrlock                0.8.3
filter_functions         1.2.3
fonttools                4.63.0
idna                     3.16
ImageIO                  2.37.3
kiwisolver               1.5.0
llvmlite                 0.47.0
matplotlib               3.10.9
numba                    0.65.1
numpy                    2.4.6
nvidia-cublas-cu12       12.9.2.10
nvidia-cuda-nvrtc-cu12   12.9.86
nvidia-cuda-runtime-cu12 12.9.79
nvidia-curand-cu12       10.3.10.19
nvidia-cusolver-cu12     11.7.5.82
nvidia-cusparse-cu12     12.5.10.65
nvidia-nvjitlink-cu12    12.9.86
opt_einsum               3.4.0
packaging                26.2
pillow                   12.2.0
pip                      26.1.1
pylatexenc               2.10
pyparsing                3.3.2
python-dateutil          2.9.0.post0
qiskit                   2.4.1
qopt                     1.3.5
qutip                    5.3.0
requests                 2.34.2
rustworkx                0.17.1
scipy                    1.17.1
setuptools               82.0.1
six                      1.17.0
sparse                   0.18.0
stevedore                5.8.0
tqdm                     4.67.3
typing_extensions        4.15.0
urllib3                  2.7.0
wheel                    0.47.0

You may find it useful to use a virtual environment on WSL:Ubuntu, accessed via the terminal.
