@echo off
REM Script para Windows para iniciar o projeto com Docker

echo ==========================================
echo   Churn Prediction Dashboard - Docker
echo ==========================================
echo.

REM Verificar se Docker está rodando
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Docker nao esta rodando!
    echo Inicie o Docker Desktop e tente novamente.
    exit /b 1
)

echo [OK] Docker detectado
echo.

REM Processar comando
if "%1"=="" goto usage
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="logs" goto logs
if "%1"=="build" goto build
if "%1"=="clean" goto clean
if "%1"=="shell" goto shell
if "%1"=="test" goto test
goto usage

:start
echo Iniciando containers...
docker-compose up -d
echo.
echo [OK] Dashboard rodando em: http://localhost:8501
echo [OK] Jupyter Lab rodando em: http://localhost:8888
echo.
echo Para ver logs: run.bat logs
echo Para parar: run.bat stop
goto end

:stop
echo Parando containers...
docker-compose down
echo [OK] Containers parados
goto end

:restart
echo Reiniciando containers...
docker-compose restart
echo [OK] Containers reiniciados
goto end

:logs
docker-compose logs -f
goto end

:build
echo Reconstruindo imagens...
docker-compose build --no-cache
echo [OK] Build concluido
goto end

:clean
echo Limpando containers, volumes e imagens...
docker-compose down -v
docker system prune -f
echo [OK] Limpeza concluida
goto end

:shell
echo Abrindo shell no container dashboard...
docker-compose exec dashboard /bin/bash
goto end

:test
echo Executando testes...
docker-compose exec dashboard python test_pipeline.py
goto end

:usage
echo Uso: run.bat {start^|stop^|restart^|logs^|build^|clean^|shell^|test}
echo.
echo Comandos:
echo   start   - Iniciar containers
echo   stop    - Parar containers
echo   restart - Reiniciar containers
echo   logs    - Ver logs em tempo real
echo   build   - Reconstruir imagens Docker
echo   clean   - Limpar containers e volumes
echo   shell   - Abrir shell no container
echo   test    - Executar testes
goto end

:end
