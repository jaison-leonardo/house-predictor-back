#!/bin/bash

echo "🐳 Construyendo imagen Docker..."

# Construir la imagen
docker build -t house-predictor-backend .

echo "✅ Imagen construida exitosamente"

# Opcional: ejecutar contenedor localmente para prueba
echo "🚀 ¿Quieres ejecutar el contenedor localmente? (y/n)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "🏃‍♂️ Ejecutando contenedor..."
    docker run -p 5000:10000 --name house-predictor-test house-predictor-backend
else
    echo "📦 Imagen lista para deploy"
