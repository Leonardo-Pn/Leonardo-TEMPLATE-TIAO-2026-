import json
import boto3
import os
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
import numpy as np
import cv2

# Importa o detector
from yolo_detector import YOLODetector

# Configurações
S3_BUCKET = os.getenv('S3_BUCKET', 'spaceguard-images')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'SpaceGuardAlerts')
MODEL_PATH = os.getenv('MODEL_PATH', '/tmp/yolo11n.pt')

# Inicializa clientes AWS
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')

# Inicializa o detector (cache global para reuso em múltiplas invocações)
_detector = None

def get_detector():
    """Singleton para o detector YOLO"""
    global _detector
    if _detector is None:
        # Baixa o modelo do S3 se necessário
        model_local_path = MODEL_PATH
        if not os.path.exists(model_local_path):
            try:
                s3.download_file(
                    os.getenv('MODEL_BUCKET', 'spaceguard-models'),
                    'yolo11n.pt',
                    model_local_path
                )
                print(f"Modelo baixado do S3 para {model_local_path}")
            except Exception as e:
                print(f"Erro ao baixar modelo: {e}, usando caminho local")
        
        _detector = YOLODetector(model_local_path)
    return _detector

def lambda_handler(event, context):
    """
    Função principal do Lambda
    
    Eventos esperados:
    1. Evento S3: Nova imagem enviada para o bucket
    2. Evento API Gateway: Requisição POST com imagem em base64
    """
    print(f"Evento recebido: {json.dumps(event, default=str)[:500]}...")
    
    try:
        # Determina o tipo de evento
        if 'Records' in event and event['Records'][0].get('eventSource') == 'aws:s3':
            # Evento S3
            return handle_s3_event(event)
        elif 'body' in event:
            # Evento API Gateway
            return handle_api_event(event)
        elif 'image_base64' in event:
            # Evento direto (invocação síncrona)
            return handle_direct_event(event)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Tipo de evento não reconhecido'})
            }
            
    except Exception as e:
        print(f"Erro no processamento: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_s3_event(event):
    """Processa evento do S3"""
    results = []
    
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        # Baixa a imagem do S3
        response = s3.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        
        # Processa a imagem
        detections, processed_image = process_image(image_data)
        
        # Salva imagem processada no S3
        processed_key = f"processed/{key.split('/')[-1]}"
        _, buffer = cv2.imencode('.jpg', processed_image)
        s3.put_object(
            Bucket=bucket,
            Key=processed_key,
            Body=buffer.tobytes()
        )
        
        # Verifica alertas
        alert_triggered = check_alerts(detections)
        
        if alert_triggered:
            save_alert(detections, bucket, key, processed_key)
            send_alert_notification(detections, bucket, key)
        
        # Salva metadados no DynamoDB
        save_metadata(detections, bucket, key, processed_key, alert_triggered)
        
        results.append({
            'source_key': key,
            'processed_key': processed_key,
            'detections': detections,
            'alert_triggered': alert_triggered
        })
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': len(results),
            'results': results
        })
    }

def handle_api_event(event):
    """Processa evento do API Gateway"""
    try:
        body = json.loads(event['body'])
        image_base64 = body.get('image_base64')
        
        if not image_base64:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Campo image_base64 é obrigatório'})
            }
        
        # Decodifica a imagem
        image_bytes = base64.b64decode(image_base64)
        detections, processed_image = process_image(image_bytes)
        
        # Converte para base64 para retorno
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'detections': detections,
                'processed_image': processed_b64,
                'alert_triggered': check_alerts(detections)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_direct_event(event):
    """Processa evento direto (invocação síncrona)"""
    image_base64 = event.get('image_base64')
    if not image_base64:
        return {'error': 'image_base64 não fornecido'}
    
    image_bytes = base64.b64decode(image_base64)
    detections, processed_image = process_image(image_bytes)
    
    _, buffer = cv2.imencode('.jpg', processed_image)
    processed_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    return {
        'detections': detections,
        'processed_image': processed_b64,
        'alert_triggered': check_alerts(detections)
    }

def process_image(image_data):
    """Processa a imagem com YOLO"""
    # Converte para numpy array
    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    
    # Detecta
    detector = get_detector()
    detections = detector.detect(image_array, conf_threshold=0.5)
    
    # Anota
    annotated = detector.annotate_image(image_array, detections)
    
    return detections, annotated

def check_alerts(detections):
    """Verifica se alguma detecção é crítica"""
    critical_labels = ['fire', 'smoke']
    for det in detections:
        if det['label'] in critical_labels and det['confidence'] > 0.6:
            return True
    return False

def save_alert(detections, bucket, key, processed_key):
    """Salva alerta no DynamoDB"""
    table = dynamodb.Table(DYNAMODB_TABLE)
    table.put_item(
        Item={
            'alert_id': f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{key}",
            'timestamp': datetime.utcnow().isoformat(),
            'source': f"s3://{bucket}/{key}",
            'processed_image': f"s3://{bucket}/{processed_key}",
            'detections': detections,
            'severity': 'HIGH' if any(d['label'] == 'fire' and d['confidence'] > 0.7 for d in detections) else 'MEDIUM',
            'status': 'PENDING'
        }
    )

def save_metadata(detections, bucket, key, processed_key, alert_triggered):
    """Salva metadados no DynamoDB"""
    table = dynamodb.Table(f"{DYNAMODB_TABLE}Metadata")
    table.put_item(
        Item={
            'image_id': key,
            'timestamp': datetime.utcnow().isoformat(),
            'bucket': bucket,
            'processed_key': processed_key,
            'num_detections': len(detections),
            'alert_triggered': alert_triggered,
            'labels': [d['label'] for d in detections]
        }
    )

def send_alert_notification(detections, bucket, key):
    """Envia notificação via SNS"""
    try:
        topic_arn = os.getenv('SNS_TOPIC_ARN')
        if not topic_arn:
            print("SNS_TOPIC_ARN não configurado, pulando notificação")
            return
        
        message = {
            'alert_type': 'critical' if any(d['label'] == 'fire' for d in detections) else 'warning',
            'source': f"s3://{bucket}/{key}",
            'detections': detections,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        sns.publish(
            TopicArn=topic_arn,
            Message=json.dumps(message),
            Subject=f"SpaceGuard Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        print("Notificação SNS enviada")
        
    except Exception as e:
        print(f"Erro ao enviar notificação SNS: {e}")