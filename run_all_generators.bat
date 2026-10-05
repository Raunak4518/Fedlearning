@echo off
echo ========================================
echo Starting Generator Comparison Suite
echo ========================================



echo.
echo [4/7] Running GAN (Baseline)...
python GeFL_GAN.py --config configs/cifar10_lt.yaml --name gefl_gan_cifar10lt_baseline

echo.
echo [5/7] Running GAN (Proposed)...
python GeFL_GAN.py --config configs/cifar10_lt_proposed.yaml --name gefl_gan_cifar10lt_proposed

echo.
echo [6/7] Running DDPM (Baseline)...
python GeFL_DDPM.py --config configs/cifar10_lt.yaml --name gefl_ddpm_cifar10lt_baseline

echo.
echo [7/7] Running DDPM (Proposed)...
python GeFL_DDPM.py --config configs/cifar10_lt_proposed.yaml --name gefl_ddpm_cifar10lt_proposed

echo.
echo ========================================
echo All Generator Runs Completed!
echo ========================================
