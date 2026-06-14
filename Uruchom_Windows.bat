@echo off
cd /d "%~dp0"
title FaceMorph
echo Uruchamiam FaceMorph... okno pojawi sie za chwile.
python FaceMorph.py
if errorlevel 1 (
  echo.
  echo ============================================================
  echo  Cos nie zadzialalo. Najpierw zainstaluj zaleznosci:
  echo     pip install -r requirements.txt
  echo  a potem uruchom ten plik ponownie.
  echo ============================================================
  echo.
  pause
)
