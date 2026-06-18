@echo off
echo ===================================================
echo Mengunduh dan Menginstal Dependensi Python...
echo ===================================================
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Gagal menginstal dependensi. Pastikan Python dan pip sudah ditambahkan ke PATH.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo Mengunduh dataset NLTK (punkt & stopwords)...
echo ===================================================
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo.
echo ===================================================
echo Instalasi Selesai!
echo Anda sekarang bisa menjalankan run_cli.bat atau run_web.bat.
echo ===================================================
pause
