@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 || exit /b 1
if "%CONDA_ENV%"=="" set "CONDA_ENV=%USERPROFILE%\anaconda3\envs\vggt"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "CUDA_HOME=%CONDA_ENV%\Library"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin;%PATH%"
set "LIB=%CONDA_ENV%\Library\lib;%LIB%"

where cl || exit /b 1
where nvcc || exit /b 1
"%CONDA_ENV%\python.exe" -m pip install --no-build-isolation %*
