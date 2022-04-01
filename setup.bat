@echo off

pip install -r requirements.txt

cd %~dp0
powershell -Command "Invoke-WebRequest https://fox-gieg.com/patches/github/n1ckfg/informative-drawings/models/checkpoints.zip -OutFile checkpoints.zip"
powershell Expand-Archive checkpoints.zip -DestinationPath .
del checkpoints.zip

@pause