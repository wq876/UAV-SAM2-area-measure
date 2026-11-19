@echo off
chcp 65001 >nul
title UAV Area Measure
rem 命令行轻量版：菜单式选照片，或把照片/文件夹拖到本文件上
cd /d "%~dp0uav_area"

set "PY=D:\Miniconda_3\envs\gee\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" run_measure.py %*
if errorlevel 1 pause
