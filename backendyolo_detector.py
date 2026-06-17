import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch

class YOLODetector:
    """
    Classe wrapper para YOLOv8 com funcionalidades específicas para detecção
    de anomalias em imagens de satélite.
    """
    
    def __init__(self, model_path: str = 'yolo11n.pt', device: str = 'cpu'):
        """
        Inicializa o detector YOLO
        
        Args:
            model_path: Caminho para o arquivo de pesos (.pt)
            device: 'cpu' ou 'cuda' (GPU)
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.classes = []
        self.load_model()
        
    def load_model(self):
        """Carrega o modelo YOLO do disco"""
        try:
            self.model = YOLO(self.model_path)
            self.classes = self.model.names
            print(f"✅ Modelo YOLO carregado: {self.model_path}")
            print(f"📊 Classes: {self.classes}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5, 
               iou_threshold: float = 0.45) -> List[Dict]:
        """
        Realiza detecção em uma imagem
        
        Args:
            image: Array numpy da imagem (H, W, C)
            conf_threshold: Limiar de confiança (0-1)
            iou_threshold: Limiar IoU para NMS
            
        Returns:
            Lista de detecções com label, confiança e bounding box
        """
        if self.model is None:
            self.load_model()
            
        # Realiza a inferência
        results = self.model(image, conf=conf_threshold, iou=iou_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Coordenadas (xyxy)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.classes[cls] if cls in self.classes else f"class_{cls}"
                    
                    detections.append({
                        'label': label,
                        'confidence': round(conf, 3),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': cls
                    })
        
        # Ordena por confiança (decrescente)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections
    
    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Desenha bounding boxes e labels na imagem
        
        Args:
            image: Array numpy da imagem original
            detections: Lista de detecções do método detect()
            
        Returns:
            Imagem anotada
        """
        img_copy = image.copy()
        
        # Cores para diferentes classes
        color_map = {
            'fire': (0, 0, 255),      # Vermelho
            'smoke': (255, 165, 0),   # Laranja
            'deforestation': (0, 255, 0),  # Verde
            'flood': (0, 255, 255),   # Amarelo
            'default': (255, 0, 0)    # Azul
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            # Escolhe a cor
            color = color_map.get(label, color_map['default'])
            
            # Desenha retângulo
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
            
            # Prepara o texto
            text = f"{label} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Calcula tamanho do texto para background
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Desenha fundo do texto
            cv2.rectangle(img_copy, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            
            # Desenha texto
            cv2.putText(img_copy, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        return img_copy
    
    def detect_and_annotate(self, image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[List[Dict], np.ndarray]:
        """
        Combina detecção e anotação em um único método
        """
        detections = self.detect(image, conf_threshold)
        annotated = self.annotate_image(image, detections)
        return detections, annotated
    
    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'num_classes': len(self.classes),
            'classes': self.classes
        }

# Teste rápido
if __name__ == "__main__":
    # Cria um detector
    detector = YOLODetector('yolo11n.pt')
    
    # Cria uma imagem fake para teste
    fake_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Detecta
    detections = detector.detect(fake_image)
    print(f"Detecções: {len(detections)}")
    
    # Anota
    annotated = detector.annotate_image(fake_image, detections)
    print(f"Imagem anotada: {annotated.shape}")