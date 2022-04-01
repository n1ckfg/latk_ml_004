@echo off

pip install -r requirements-win.txt
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

cd %~dp0
powershell -Command "Invoke-WebRequest https://fox-gieg.com/patches/github/n1ckfg/informative-drawings/models/checkpoints.zip -OutFile checkpoints.zip"
powershell Expand-Archive checkpoints.zip -DestinationPath .
del checkpoints.zip

@pause