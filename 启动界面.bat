@echo off
chcp 65001 >nul
title UAV Area Measure - GUI
rem 用脚本自身所在目录，clone 到任何位置都能跑
cd /d "%~dp0uav_area"

rem 优先用本机 conda 环境里的 python，找不到就退回 PATH 里的 python
set "PY=D:\Miniconda_3\envs\gee\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" app_gui.py %*
if errorlevel 1 pause
