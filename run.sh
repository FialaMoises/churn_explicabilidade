#!/bin/bash
# Script para iniciar o projeto com Docker

echo "=========================================="
echo "  Churn Prediction Dashboard - Docker"
echo "=========================================="
echo ""

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "ERRO: Docker não está instalado!"
    echo "Instale Docker em: https://docs.docker.com/get-docker/"
    exit 1
fi

# Verificar se docker-compose está instalado
if ! command -v docker-compose &> /dev/null; then
    echo "ERRO: docker-compose não está instalado!"
    echo "Instale docker-compose em: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker detectado"
echo "✓ docker-compose detectado"
echo ""

# Opções
case "$1" in
    start)
        echo "Iniciando containers..."
        docker-compose up -d
        echo ""
        echo "✓ Dashboard rodando em: http://localhost:8501"
        echo "✓ Jupyter Lab rodando em: http://localhost:8888"
        echo ""
        echo "Para ver logs: ./run.sh logs"
        echo "Para parar: ./run.sh stop"
        ;;

    stop)
        echo "Parando containers..."
        docker-compose down
        echo "✓ Containers parados"
        ;;

    restart)
        echo "Reiniciando containers..."
        docker-compose restart
        echo "✓ Containers reiniciados"
        ;;

    logs)
        docker-compose logs -f
        ;;

    build)
        echo "Reconstruindo imagens..."
        docker-compose build --no-cache
        echo "✓ Build concluído"
        ;;

    clean)
        echo "Limpando containers, volumes e imagens..."
        docker-compose down -v
        docker system prune -f
        echo "✓ Limpeza concluída"
        ;;

    shell)
        echo "Abrindo shell no container dashboard..."
        docker-compose exec dashboard /bin/bash
        ;;

    test)
        echo "Executando testes..."
        docker-compose exec dashboard python test_pipeline.py
        ;;

    *)
        echo "Uso: ./run.sh {start|stop|restart|logs|build|clean|shell|test}"
        echo ""
        echo "Comandos:"
        echo "  start   - Iniciar containers"
        echo "  stop    - Parar containers"
        echo "  restart - Reiniciar containers"
        echo "  logs    - Ver logs em tempo real"
        echo "  build   - Reconstruir imagens Docker"
        echo "  clean   - Limpar containers e volumes"
        echo "  shell   - Abrir shell no container"
        echo "  test    - Executar testes"
        exit 1
        ;;
esac
