# bev_vqa

La logica del framework è racchiusa nel package bev_vqa:

## MODELS

Qua sono racchiusi i modelli: projector e vlm

### projector.py

Comprime la mappa BEV [128, 200, 200] nello spazio token del LLM.

DeeperConvProjector, ProjectorConfig, get_spatial_pool_shape

### vlm.py

Assembla proiettore, LLM (Qwen2.5) e LoRA; gestisce l'iniezione multimodale.

BEVVLM, prepare_multimodal_inputs

### dataset.py

Caricamento dataset JSON, cache .pkl dei campioni, risoluzione dei tensori .pt.

BaseBEVDataset, BEVQADataset, build_tokenizer

### collate.py

Batching con padding dinamico per training (right-pad) ed evaluation (left-pad per generazione).

collate_train, collate_eval

### sampler.py

Campionamento bilanciato per categoria per azzerare i bias statistici del dataset.

CategoryBalancedSampler, DriveLMCategoryBalancedSampler

### losses.py

Loss SOTA avanzata: penalizzazione smooth per errori numerici e pesatura classi rare.

NuScenesBoostLoss, compute_gaussian_count_distribution

### metrics.py

Normalizzazione risposte e metriche di testo generativo con COCO captioning / NLTK.

normalize_answer, compute_nlg_metrics

### visualization.py

Dashboard di debug visuale a 150 DPI (6 telecamere surround + heatmap BEV L2 con marker ego).

build_multimodal_debug_figure, compute_bev_l2_heatmap

### io.py

Caricamento ottimizzato con fallback automatico tra cartelle val/ e train/.

load_bev_features, get_default_paths
