import os
# Silenciar advertencias de fuentes en Linux Mint
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

import cv2
from ultralytics import YOLO

def main():
    # Cargar los pesos entrenados mediante su ruta absoluta
    model = YOLO("/home/geovani/TMR/latas_tmr_v2/weights/best.pt")

    # Usar el puerto 2 para la cámara USB externa (/dev/video2)
    puerto_camara = 2
    cap = cv2.VideoCapture(puerto_camara)

    if not cap.isOpened():
        print(f"Error: No se pudo acceder a la cámara en /dev/video{puerto_camara}.")
        print("Si no abre, prueba cambiando 'puerto_camara = 2' a 3 en el código.")
        return

    print("Cámara USB encendida. Presiona 'q' en la ventana para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se reciben imágenes de la cámara.")
            break

        # Inferencia con umbral de confianza al 50%
        results = model.predict(
            source=frame,
            conf=0.50,
            imgsz=512,
            verbose=False
        )

        # Dibujar las detecciones sobre la imagen
        annotated_frame = results[0].plot()
        cv2.imshow("Detección de Latas TMR - V3", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()