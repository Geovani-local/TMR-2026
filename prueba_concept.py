import cv2
from ultralytics import YOLO

def main():
    # Cargar los pesos óptimos del nuevo entrenamiento v3
    model = YOLO(
        #"/home/geovani/yolo-plate-recognition/runs/detect/latas_tmr_v3/weights/best.pt"
        "/home/geovani/TMR/latas_tmr_v2/weights/best.pt"

    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia con umbral de confianza al 50%
        results = model.predict(
            source=frame,
            conf=0.50,
            imgsz=512,
            verbose=False
        )

        annotated_frame = results[0].plot()
        cv2.imshow("Detección de Latas TMR - V3", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()