import os
import json
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import boto3
from datetime import datetime
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

# Importações locais
from yolo_detector import YOLODetector
from models import DetectionRequest, DetectionResponse, Alert
from utils import save_to_s3, save_to_dynamodb, send_alert

# Inicialização
app = FastAPI(
    title="SpaceGuard AI API",
    description="API para detecção de anomalias em imagens de satélite usando YOLOv8",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações
S3_BUCKET = os.getenv("S3_BUCKET", "spaceguard-images")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "SpaceGuardAlerts")
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo11n.pt")

# Inicializa o detector (singleton)
detector = YOLODetector(model_path=MODEL_PATH)

# Clientes AWS (mockáveis)
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv("AWS_ENDPOINT_URL", None),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

dynamodb = boto3.resource(
    'dynamodb',
    endpoint_url=os.getenv("AWS_ENDPOINT_URL", None),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Modelos Pydantic
class ImageURLRequest(BaseModel):
    url: str
    confidence_threshold: Optional[float] = 0.5

class DetectionResult(BaseModel):
    label: str
    confidence: float
    bbox: list

class ProcessResponse(BaseModel):
    success: bool
    message: str
    detections: list
    processed_image_url: Optional[str] = None
    alert_triggered: bool = False

# ==================== ENDPOINTS ====================

@app.get("/")
async def health_check():
    """Health check da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "model_loaded": detector.model is not None
    }

@app.post("/detect")
async def detect_anomalies(file: UploadFile = File(...), threshold: Optional[float] = 0.5):
    """
    Detecta anomalias em uma imagem enviada (multipart/form-data)
    """
    try:
        # Lê a imagem
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_array = np.array(image)
        
        # Detecta
        detections = detector.detect(image_array, conf_threshold=threshold)
        
        # Gera imagem anotada
        annotated = detector.annotate_image(image_array, detections)
        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Salva no S3 (em background)
        filename = f"processed/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_to_s3(s3_client, S3_BUCKET, filename, buffer.tobytes())
        
        # Verifica se precisa gerar alerta
        alert_triggered = any(d['label'] in ['fire', 'smoke'] for d in detections)
        
        if alert_triggered:
            alert_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'detections': detections,
                'image_key': filename,
                'severity': 'HIGH' if any(d['label'] == 'fire' and d['confidence'] > 0.7 for d in detections) else 'MEDIUM'
            }
            save_to_dynamodb(dynamodb, DYNAMODB_TABLE, alert_data)
            send_alert(alert_data)  # SNS ou Webhook
        
        return DetectionResponse(
            success=True,
            detections=detections,
            processed_image=img_b64,
            alert_triggered=alert_triggered,
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/url")
async def detect_from_url(request: ImageURLRequest):
    """
    Detecta anomalias em uma imagem a partir de uma URL
    """
    import requests
    try:
        response = requests.get(request.url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)
        
        detections = detector.detect(image_array, conf_threshold=request.confidence_threshold)
        
        return {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar URL: {str(e)}")

@app.get("/alerts")
async def get_alerts(limit: int = 10):
    """
    Retorna os últimos alertas registrados
    """
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        response = table.scan(
            Limit=limit,
            ScanIndexForward=False
        )
        return {
            "success": True,
            "alerts": response.get('Items', [])
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "alerts": []
        }

@app.get("/model/info")
async def get_model_info():
    """Informações sobre o modelo carregado"""
    return {
        "model_path": MODEL_PATH,
        "num_classes": len(detector.model.names) if detector.model else 0,
        "classes": detector.model.names if detector.model else {}
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )