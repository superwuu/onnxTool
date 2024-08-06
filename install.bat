@echo off
setlocal enabledelayedexpansion

:: 创建必要的目录
if not exist build\include mkdir build\include
if not exist build\lib mkdir build\lib

:: 复制所有的 .h 文件
xcopy /Y /I *.h build\include\
xcopy /Y /I depend\*.h build\include\
xcopy /Y /I code\*.h build\include\

:: 复制所有的 .dll和.lib 文件
xcopy /Y /I depend\*.dll build\lib\
xcopy /Y /I depend\*.lib build\lib\
xcopy /Y /I build\Release\*.dll build\lib\
xcopy /Y /I build\Release\*.lib build\lib\

echo Files have been copied successfully.
pause