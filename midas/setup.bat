@echo off

cd %~dp0
cd model
powershell -Command "Invoke-WebRequest https://fox-gieg.com/patches/github/n1ckfg/MiDaS/model/model.zip -OutFile model.zip"
powershell Expand-Archive model.zip -DestinationPath .
del model.zip

@pause