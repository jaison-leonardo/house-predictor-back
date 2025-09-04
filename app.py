from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import sklearn
import xgboost

# Crear la app Flask
app = Flask(__name__)

# Configurar CORS para permitir requests desde Vercel
CORS(app, origins=["*"])  # En producción, especifica tu dominio de Vercel

# Variables globales para modelos
pipeline_completo = None
metadata = None


def load_models():
    global pipeline_completo, metadata
    try:
        pipeline_path = os.path.join(os.path.dirname(__file__), 'xgb_pipeline.pkl')
        metadata_path = os.path.join(os.path.dirname(__file__), 'model_metadata.pkl')

        print(f"🔍 Intentando cargar modelos...")
        print(f"📍 Directorio actual: {os.path.dirname(__file__)}")
        print(f"📄 Archivos disponibles: {os.listdir(os.path.dirname(__file__))}")
        print(f"📊 Versión sklearn actual: {sklearn.__version__}")
        print(f"🚀 Versión xgboost actual: {xgboost.__version__}")

        # Cargar pipeline completo
        if os.path.exists(pipeline_path):
            try:
                pipeline_completo = joblib.load(pipeline_path)
                print("✅ Pipeline completo cargado exitosamente")
            except Exception as e:
                print(f"❌ Error cargando pipeline: {e}")
                # Intentar con pickle como alternativa
                try:
                    with open(pipeline_path, 'rb') as f:
                        pipeline_completo = pickle.load(f)
                    print("✅ Pipeline cargado con pickle como alternativa")
                except Exception as e2:
                    print(f"❌ Error cargando pipeline con pickle: {e2}")
                    pipeline_completo = None

        # Cargar metadata si existe
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                print(f"✅ Metadata cargada - Entrenado con sklearn {metadata.get('sklearn_version', 'desconocida')}")
            except Exception as e:
                print(f"⚠️ No se pudo cargar metadata: {e}")
                metadata = None

    except Exception as e:
        print(f"❌ Error general cargando modelos: {e}")


# Cargar modelos al inicializar
load_models()


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "🏠 API de Predicción de Precios de Casas",
        "version": "2.0.0",
        "status": "active",
        "pipeline_loaded": pipeline_completo is not None,
        "metadata_loaded": metadata is not None,
        "current_versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__
        },
        "training_versions": {
            "sklearn": metadata.get('sklearn_version', 'desconocida') if metadata else 'no disponible',
            "xgboost": metadata.get('xgboost_version', 'desconocida') if metadata else 'no disponible'
        } if metadata else None,
        "endpoints": {
            "predict": "/api/predict (POST)",
            "health": "/api/health (GET)"
        }
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Manejar preflight CORS
    if request.method == 'OPTIONS':
        return '', 200

    try:
        if pipeline_completo is None:
            return jsonify({
                "error": "Pipeline no disponible. Contacta al administrador.",
                "status": "error"
            }), 500

        # Obtener datos del request
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No se enviaron datos para la predicción",
                "status": "error"
            }), 400

        # Validar campos requeridos
        required_fields = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
            'sqft_living15', 'sqft_lot15'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Campos faltantes: {', '.join(missing_fields)}",
                "status": "error"
            }), 400

        print(f"📊 Datos recibidos: {data}")

        # Crear DataFrame con los datos recibidos
        df_pred = pd.DataFrame([data])

        # Asegurar que tiene todas las columnas necesarias (según metadata)
        if metadata and 'features' in metadata:
            expected_features = metadata['features']
            for col in expected_features:
                if col not in df_pred.columns:
                    df_pred[col] = 0
            # Reordenar columnas según el orden de entrenamiento
            df_pred = df_pred[expected_features]

        # Hacer predicción usando el pipeline completo
        log_prediction = pipeline_completo.predict(df_pred)[0]

        # Convertir de log a precio real (ya que usamos log1p en el entrenamiento)
        precio_real = np.expm1(log_prediction)

        # Asegurar que la predicción sea positiva
        precio_real = max(0, float(precio_real))

        print(f"✅ Predicción exitosa: ${precio_real:,.2f}")

        return jsonify({
            "precio_estimado": round(precio_real, 2),
            "status": "success",
            "method": "pipeline_completo",
            "log_prediction": float(log_prediction),
            "timestamp": datetime.now().isoformat(),
            "datos_procesados": data,
            "model_info": {
                "features_used": len(df_pred.columns),
                "training_r2": metadata.get('metrics', {}).get('test_r2') if metadata else None
            }
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error en predicción: {error_msg}")

        # Mensajes de error más específicos
        if "ColumnTransformer" in error_msg:
            error_msg = f"Error de compatibilidad de versiones. Modelo entrenado con sklearn {metadata.get('sklearn_version') if metadata else 'desconocida'}, servidor usando {sklearn.__version__}"
        elif "feature" in error_msg.lower():
            error_msg = "Error en las características enviadas. Verifica que todos los campos sean correctos."

        return jsonify({
            "error": f"Error en predicción: {error_msg}",
            "status": "error",
            "debug_info": {
                "current_sklearn": sklearn.__version__,
                "training_sklearn": metadata.get('sklearn_version') if metadata else 'desconocida',
                "pipeline_loaded": pipeline_completo is not None
            }
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    status_info = {
        "status": "healthy" if pipeline_completo is not None else "degraded",
        "pipeline_loaded": pipeline_completo is not None,
        "metadata_loaded": metadata is not None,
        "current_versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "timestamp": datetime.now().isoformat(),
        "server": "Render"
    }

    if metadata:
        status_info["training_info"] = {
            "sklearn_version": metadata.get('sklearn_version'),
            "xgboost_version": metadata.get('xgboost_version'),
            "features_count": len(metadata.get('features', [])),
            "model_performance": metadata.get('metrics', {}),
            "training_date": metadata.get('fecha_entrenamiento')
        }

    return jsonify(status_info)


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Endpoint adicional para información detallada del modelo"""
    if not metadata:
        return jsonify({
            "error": "Metadata del modelo no disponible",
            "status": "error"
        }), 404

    return jsonify({
        "model_metadata": metadata,
        "status": "success"
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint no encontrado",
        "status": "error",
        "available_endpoints": ["/", "/api/predict", "/api/health", "/api/model-info"]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Error interno del servidor",
        "status": "error"
    }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)