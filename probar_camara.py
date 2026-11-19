import os
# Silenciar advertencias de la terminal sobre fuentes
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

import cv2
from ultralytics import YOLO

def main():
    # 1. Cargar el modelo entrenado
    model = YOLO("best.pt")

    # 2. Configurar la cámara USB externa (puerto 2)
    PUERTO_CAMARA = 2  
    cap = cv2.VideoCapture(PUERTO_CAMARA)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara en el puerto {PUERTO_CAMARA}.")
        return

    print("¡Cámara USB encendida con éxito! Presiona la tecla 'q' en la ventana para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se reciben imágenes de la cámara.")
            break

        # 3. Inferencia de YOLOv8 sobre la cámara en vivo
        results = model.predict(source=frame, conf=0.50, imgsz=512, verbose=False)

        # 4. Dibujar las cajas delimitadoras
        annotated_frame = results[0].plot()

        # 5. Mostrar el resultado en pantalla
        cv2.imshow("Prueba Camara USB - Deteccion TMR", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()