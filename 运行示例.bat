@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  lecture-notes 运行示例
echo ============================================
echo.
echo [1/2] 合成 demo 素材（ffmpeg lavfi，无需外部文件）
python make_demo.py
if errorlevel 1 goto fail
echo.
echo [2/2] 用 demo 跑一遍完整流程（未配置模型密钥时会跳过笔记）
python lecture_notes.py demo\demo.mp4 -t demo\demo.srt --title 演示讲座 -o 输出测试
if errorlevel 1 goto fail
echo.
echo 完成，结果在：输出测试\
dir /b 输出测试
pause
exit /b 0

:fail
echo.
echo !! 执行失败，请检查上面的报错（多半是缺 ffmpeg 或缺依赖包）
pause
exit /b 1
