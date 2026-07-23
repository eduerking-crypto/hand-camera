# Hand Camera - Pizarra Inteligente

Dibuja en una pizarra virtual usando **gestos de tu mano** frente a la cámara web.

## Gestos

| Gesto | Accion |
|-------|--------|
| **Indice alzado** | Mover cursor |
| **Pinza (pulgar + indice juntos)** | Dibujar / escribir |
| **2 dedos (paz)** | Siguiente herramienta |
| **5 dedos + pinza** | Limpiar pizarra |
| **Senal de paz (2 dedos, sin pinza)** | Cambiar herramienta |

## Herramientas

- **Lapiz** - trazo fino negro
- **Marcador** - trazo grueso
- **Resaltador** - trazo amarillo ancho
- **Borrador** - borra por areas

## Instalacion

```bash
pip install -r requirements.txt
python download_models.py
python main.py
```

## Controles teclado

- `ESC` - Salir
- `C` - Limpiar pizarra
- `S` - Guardar captura (pizarra.png)
