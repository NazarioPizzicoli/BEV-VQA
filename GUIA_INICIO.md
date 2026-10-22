# Guía Rápida de Inicio: BEV-VLM

En tu carpeta `~/nazario` ya dispones de los datos y pesos esenciales:
- `checkpoints/`: Modelos entrenados (`multiscale_peak` + `2d_polar_metric` y `stage1`)
- `bev_features/`: Tensores Bird's-Eye-View extraídos (`val/` y `train/`)
- `vqa_datasets/`: Anotaciones unificadas en formato JSON

---

## Paso 1: Clonar el Repositorio de GitHub (ya eres colaborador)

```bash
cd ~/nazario
git clone https://github.com/NazarioPizzicoli/BEV-VQA.git
cd BEV-VQA
```

---

## Paso 2: Vincular Checkpoints y Configurar Rutas

En lugar de copiar gigabytes de datos, crea un enlace simbólico a los checkpoints y define el archivo de configuración `.env`:

```bash
# 1. Enlazar la carpeta de checkpoints dentro del repositorio
rm -rf checkpoints && ln -s ~/nazario/checkpoints checkpoints

# 2. Configurar las rutas para Docker mediante archivo .env
cat << 'EOF' > .env
HOST_DATA_DIR=/home/robesafe/nazario
BEV_VQA_BEV_DIR=/data/bev_features
BEV_VQA_DATASETS_DIR=/data/vqa_datasets/unified
EOF
```

> **Nota:** Si tus archivos JSON se encuentran directamente en `~/nazario/vqa_datasets/` (en vez de dentro de `unified/`), ajusta la última línea del `.env` a:
> `BEV_VQA_DATASETS_DIR=/data/vqa_datasets`

---

## Paso 3: Ejecución con Docker (Recomendado)

El contenedor está completamente preparado con **CUDA 12.1 + PyTorch 2.4** y todas las librerías preinstaladas.

### 1. Construir la imagen Docker
```bash
docker compose build
```

### 2. Verificar GPU y entorno
Comprueba que PyTorch detecta tu GPU y que la arquitectura carga correctamente:
```bash
docker compose run --rm bev-vlm python scripts/verify_docker_environment.py
```

### 3. Prueba interactiva (Inferencia real y visualización)

* **Menú interactivo completo (seleccionar preguntas preindexadas [1-N], hacer preguntas personalizadas con [C] o cambiar de escena con [N]):**
  ```bash
  docker compose run --rm interactive --interactive
  ```

* **Prueba rápida en una línea (escena aleatoria con visualización de cámaras y BEV):**
  ```bash
  docker compose run --rm interactive
  ```
  *(O haciendo una pregunta libre directamente: `docker compose run --rm interactive --random --question "¿Es seguro girar a la izquierda?" --terminal-viz`)*

### 4. Evaluación DriveLM con LLM-as-a-Judge
Ejecuta la evaluación generativa (BLEU-4, CIDEr, METEOR) combinada con el evaluador LLM local (`Qwen2.5-3B-Instruct`):

* **Prueba rápida (5 muestras para validar):**
  ```bash
  docker compose run --rm bev-vlm python scripts/evaluate_drivelm_nlg.py \
      --checkpoint checkpoints/ablation_pos_embed/2d_polar_metric/drivelm/best_model \
      --num-samples 5 \
      --llm-judge \
      --judge-samples 5
  ```

* **Evaluación completa preconfigurada (100 muestras):**
  ```bash
  docker compose run --rm eval-drivelm
  ```

### 5. Ejecutar la suite de tests unitarios
Valida los 74 tests de proyectores, embeddings polares y pipeline:
```bash
docker compose run --rm test
```