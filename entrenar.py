from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    # Ruta correcta hacia el data.yaml del dataset extraído
    model.train(
        data="/home/geovani/TMR/dataset/data.yaml",
        epochs=80,
        imgsz=512,
        batch=8,
        device="cpu",
        workers=4,
        name="latas_tmr_v3"   # Le ponemos v3 para no sobreescribir la v2
    )

if __name__ == "__main__":
    main()