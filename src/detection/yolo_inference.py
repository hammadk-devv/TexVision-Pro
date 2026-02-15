"""
YOLOv8 Inference Module for Real-time Defect Detection
Optimized for both PC and Raspberry Pi deployment.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
import time


class YOLODetector:
    """
    YOLOv8 detector for fabric defect detection
    Supports both PyTorch and ONNX models for flexibility
    """
    
    def __init__(
        self, 
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to model file (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', or device index)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.class_names = None
        
        # Performance metrics
        self.inference_times = []
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model based on file extension"""
        if self.model_path.suffix == '.pt':
            self._load_pytorch_model()
        elif self.model_path.suffix == '.onnx':
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
    
    def _load_pytorch_model(self):
        """Load PyTorch YOLOv8 model"""
        from ultralytics import YOLO
        
        print(f"Loading PyTorch model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        
        # Get class names
        self.class_names = self.model.names
        print(f"Model loaded successfully. Classes: {self.class_names}")
    
    def _load_onnx_model(self):
        """Load ONNX model for optimized inference"""
        import onnxruntime as ort
        
        print(f"Loading ONNX model: {self.model_path}")
        
        # Set providers based on device
        if self.device == 'cuda' or (isinstance(self.device, int) and self.device >= 0):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.model = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        # Get input/output details
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [output.name for output in self.model.get_outputs()]
        
        print(f"ONNX model loaded successfully")
        print(f"Input: {self.input_name}")
        print(f"Outputs: {self.output_names}")
    
    def detect(
        self, 
        image: np.ndarray,
        return_crops: bool = False
    ) -> Union[List[dict], Tuple[List[dict], List[np.ndarray]]]:
        """
        Detect defects in an image
        
        Args:
            image: Input image (BGR format)
            return_crops: If True, return cropped defect regions
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection confidence
                - class_id: class ID
                - class_name: class name
            If return_crops=True, also returns list of cropped images
        """
        start_time = time.time()
        
        if self.model_path.suffix == '.pt':
            detections = self._detect_pytorch(image)
        else:
            detections = self._detect_onnx(image)
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Crop defect regions if requested
        if return_crops:
            crops = self._crop_detections(image, detections)
            return detections, crops
        
        return detections
    
    def _detect_pytorch(self, image: np.ndarray) -> List[dict]:
        """Run detection using PyTorch model"""
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=416,
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            detection = {
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': self.class_names[int(box.cls[0])]
            }
            detections.append(detection)
        
        return detections
    
    def _detect_onnx(self, image: np.ndarray) -> List[dict]:
        """Run detection using ONNX model"""
        # Preprocess image
        input_tensor = self._preprocess_onnx(image)
        
        # Run inference
        outputs = self.model.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess outputs
        detections = self._postprocess_onnx(outputs, image.shape)
        
        return detections
    
    def _preprocess_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize to model input size (typically 320x320 or 640x640)
        self.input_size = 416  # Architecture Alignment: 416x416
        img = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _postprocess_onnx(self, outputs: List[np.ndarray], image_shape: Tuple) -> List[dict]:
        """Postprocess ONNX model outputs for YOLOv8"""
        predictions = outputs[0]  # Shape: (1, 84, 8400) or similar
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dim
            
        # YOLOv8 output: [x, y, w, h, class0, class1, ...]
        # Transpose to (8400, 84)
        predictions = predictions.T
        
        boxes = []
        scores = []
        class_ids = []
        
        img_h, img_w = image_shape[:2]
        x_factor = img_w / self.input_size
        y_factor = img_h / self.input_size
        
        for pred in predictions:
            classes_scores = pred[4:]
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]
            
            if score > self.conf_threshold:
                # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
                cx, cy, w, h = pred[0:4]
                x1 = (cx - 0.5 * w) * x_factor
                y1 = (cy - 0.5 * h) * y_factor
                x2 = (cx + 0.5 * w) * x_factor
                y2 = (cy + 0.5 * h) * y_factor
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                class_ids.append(int(class_id))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            [ [b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes ],
            scores,
            self.conf_threshold,
            self.iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': boxes[i],
                    'confidence': scores[i],
                    'class_id': class_ids[i],
                    'class_name': self.class_names[class_ids[i]]
                })
        
        return detections
    
    def _crop_detections(self, image: np.ndarray, detections: List[dict]) -> List[np.ndarray]:
        """Crop detected regions from image"""
        crops = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Crop region
            crop = image[y1:y2, x1:x2]
            crops.append(crop)
        
        return crops
    
    def visualize(
        self, 
        image: np.ndarray, 
        detections: List[dict],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on image
        
        Args:
            image: Input image
            detections: List of detections
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Define colors for different classes (BGR format)
        colors = [
            (0, 255, 0),    # Green - holes
            (0, 0, 255),    # Red - stains
            (255, 0, 0),    # Blue - misprints
            (0, 255, 255),  # Yellow - color variations
        ]
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, annotated)
        
        return annotated
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in seconds"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)
    
    def get_fps(self) -> float:
        """Get average FPS"""
        avg_time = self.get_avg_inference_time()
        if avg_time == 0:
            return 0.0
        return 1.0 / avg_time


def main():
    """Demo inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Defect Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pt or .onnx)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or video')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda, or device index)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save output')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not show window')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Load image
    image = cv2.imread(args.source)
    if image is None:
        print(f"Error: Could not load image from {args.source}")
        return
    
    # Run detection
    print("Running detection...")
    detections = detector.detect(image)
    
    # Print results
    print(f"\nDetected {len(detections)} defects:")
    for i, det in enumerate(detections):
        print(f"{i+1}. {det['class_name']}: {det['confidence']:.2f}")
    
    # Visualize
    annotated = detector.visualize(image, detections, save_path=args.save)
    
    # Show performance
    print(f"\nInference time: {detector.get_avg_inference_time()*1000:.2f} ms")
    print(f"FPS: {detector.get_fps():.2f}")
    
    # Display (optional)
    if not args.no_show:
        cv2.imshow('Detections', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
