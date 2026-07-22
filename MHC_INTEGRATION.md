# mHC Integration Viability Analysis for CodKing

**Paper**: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
**Authors**: DeepSeek-AI (Zhenda Xie et al., 20 authors)
**Date**: December 2025 / January 2026
**Analysis Date**: 2026-01-09

---

## Executive Summary

**Viability Assessment: HIGHLY RECOMMENDED (9/10)**

mHC (Manifold-Constrained Hyper-Connections) es una innovación arquitectónica de DeepSeek que **resuelve directamente problemas de estabilidad de entrenamiento en redes profundas** mediante la proyección de conexiones residuales expandidas sobre el politopo de Birkhoff (matrices doblemente estocásticas). Esta técnica es **altamente compatible** con la arquitectura de CodKing y podría mejorar significativamente:

- **Estabilidad de entrenamiento** en las 22 capas del Main Network de H-Net
- **Escalabilidad** para modelos más profundos sin gradientes que desaparecen
- **Rendimiento en tareas downstream** (+5-7% en benchmarks según paper)

**Overhead**: Solo 6.7% adicional en tiempo de entrenamiento.

---

## 1. Análisis Técnico de mHC

### 1.1 Problema que Resuelve

Las **Hyper-Connections (HC)** tradicionales expanden el stream residual de `1 → n` canales paralelos (típicamente n=4), pero esto compromete la **propiedad de mapeo de identidad** que estabiliza ResNets:

```
Residual estándar: x_{l+1} = x_l + F(x_l)  [Identity mapping preservado]
Hyper-Connection:  x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)  [Identity mapping perdido]
```

**Consecuencias de HC sin restricciones:**
- Training instability (loss spikes alrededor del step 12k)
- Gradient norm explosions
- Memory overhead significativo
- Escalabilidad limitada

### 1.2 Solución mHC

mHC **proyecta las matrices de mezcla residual sobre el politopo de Birkhoff** mediante iteración Sinkhorn-Knopp diferenciable:

```python
# Politopo de Birkhoff: matrices doblemente estocásticas
# - Todas las entradas >= 0
# - Cada fila suma 1
# - Cada columna suma 1

def sinkhorn_knopp(A, iterations=5, eps=1e-8):
    """Proyecta A al politopo de Birkhoff."""
    A = torch.exp(A)  # Asegurar no-negatividad
    for _ in range(iterations):
        A = A / (A.sum(dim=-1, keepdim=True) + eps)  # Filas suman 1
        A = A / (A.sum(dim=-2, keepdim=True) + eps)  # Columnas suman 1
    return A
```

**Ecuación de actualización mHC:**
```
x_{l+1} = H_l^{res} · x_l + H_l^{post,T} · F(H_l^{pre} · x_l)

donde H^{res} es doblemente estocástica (proyectada via Sinkhorn-Knopp)
```

### 1.3 Beneficios Cuantificados

| Métrica | Baseline | HC | mHC | Mejora |
|---------|----------|----|----|--------|
| BBH (27B) | 43.8% | 48.9% | **51.0%** | +7.2% |
| DROP | baseline | +X | **+Y** | - |
| GSM8K | baseline | +X | **+Y** | - |
| Training Stability | ✅ | ❌ (spikes) | ✅ | - |
| Gradient Norm | Smooth | Unstable | **Smooth** | - |
| Overhead | 0% | ~10% | **6.7%** | - |

---

## 2. Arquitectura CodKing - Puntos de Integración

### 2.1 Componentes Actuales

```
CodKing Architecture:
├── H-Net (94M params)
│   ├── Encoder (6M) - 4 capas Mamba-2/LSTM
│   ├── Dynamic Chunking (2M)
│   ├── Main Network (80M) - **22 capas Transformer** ← mHC aquí
│   └── Decoder (6M) - 4 capas Mamba-2
│
└── HRM (27M params)
    ├── H-module - **4 capas Transformer** ← mHC potencial
    └── L-module - **4 capas Transformer** ← mHC potencial
```

### 2.2 Puntos de Integración Prioritarios

#### **PRIORIDAD 1: H-Net Main Network (22 capas)**

El Main Network de H-Net tiene **22 capas Transformer** - exactamente donde mHC ofrece máximo beneficio:

```python
# Archivo: models/hnet/main_network.py
# Línea ~35-137: MultiHeadAttention con residual estándar

class TransformerBlock(nn.Module):
    def forward(self, x, attention_mask=None):
        # ACTUAL: Residual estándar
        x = x + self.attention(self.norm1(x), attention_mask)
        x = x + self.ff(self.norm2(x))
        return x

    # CON mHC: Hyper-Connections con restricción de manifold
    def forward_mhc(self, x_streams, attention_mask=None):
        # x_streams: [batch, n_streams, seq_len, d_model]
        H_res = self.sinkhorn_knopp(self.H_res_raw)  # Proyectar a Birkhoff
        H_pre = self.sinkhorn_knopp(self.H_pre_raw)
        H_post = self.sinkhorn_knopp(self.H_post_raw)

        x_mixed = torch.einsum('ij,bjsd->bisd', H_pre, x_streams)
        attn_out = self.attention(self.norm1(x_mixed.mean(dim=1)))

        x_new = torch.einsum('ij,bjsd->bisd', H_res, x_streams) + \
                torch.einsum('ji,bsd->bisd', H_post, attn_out.unsqueeze(1))
        return x_new
```

**Beneficio esperado:**
- Estabilidad en entrenamiento profundo
- Mejor flujo de gradientes en 22 capas
- +5-7% en performance downstream

#### **PRIORIDAD 2: HRM H-module y L-module (4 capas cada uno)**

Los módulos jerárquicos de HRM también usan residual estándar:

```python
# Archivo: models/hrm/modules.py
# Línea ~100-107: Post-Norm residual

# ACTUAL:
x = self.norm1(residual + self.dropout(out))  # Residual simple

# CON mHC:
x = self.norm1(H_res @ residual + H_post.T @ self.dropout(out))  # Manifold-constrained
```

**Beneficio esperado:**
- Mejor convergencia jerárquica entre H y L modules
- Representaciones más diversas en streams paralelos

---

## 3. Implementación Propuesta

### 3.1 Nuevo Módulo: `mhc_layers.py`

```python
"""
mHC: Manifold-Constrained Hyper-Connections para CodKing.
Basado en: https://arxiv.org/abs/2512.24880
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinkhornKnopp(nn.Module):
    """
    Proyección diferenciable al politopo de Birkhoff.
    Convierte cualquier matriz en doblemente estocástica.
    """

    def __init__(self, iterations: int = 5, eps: float = 1e-8):
        super().__init__()
        self.iterations = iterations
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Proyecta A al politopo de Birkhoff via Sinkhorn-Knopp.

        Args:
            A: Matriz de entrada [n_streams, n_streams] o [batch, n, n]
        Returns:
            Matriz doblemente estocástica
        """
        # Exponenciar para asegurar no-negatividad
        A = torch.exp(A)

        for _ in range(self.iterations):
            # Normalización de filas
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
            # Normalización de columnas
            A = A / (A.sum(dim=-2, keepdim=True) + self.eps)

        return A


class mHCResidual(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Residual Block.

    Expande el stream residual a n canales paralelos con matrices
    de mezcla proyectadas al politopo de Birkhoff.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        sinkhorn_iterations: int = 5
    ):
        """
        Args:
            d_model: Dimensión del modelo
            n_streams: Número de streams paralelos (default: 4 como en DeepSeek)
            sinkhorn_iterations: Iteraciones Sinkhorn-Knopp
        """
        super().__init__()

        self.d_model = d_model
        self.n_streams = n_streams

        # Matrices de mezcla (aprendibles, antes de proyección)
        self.H_res_raw = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.H_pre_raw = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.H_post_raw = nn.Parameter(torch.zeros(n_streams, n_streams))

        # Inicialización: cerca de identidad
        nn.init.eye_(self.H_res_raw)
        nn.init.eye_(self.H_pre_raw)
        nn.init.eye_(self.H_post_raw)

        # Proyección Sinkhorn-Knopp
        self.sinkhorn = SinkhornKnopp(iterations=sinkhorn_iterations)

        # Proyección de entrada/salida para expandir/contraer streams
        self.expand = nn.Linear(d_model, d_model * n_streams)
        self.contract = nn.Linear(d_model * n_streams, d_model)

    def expand_to_streams(self, x: torch.Tensor) -> torch.Tensor:
        """Expande entrada a múltiples streams."""
        # x: [batch, seq_len, d_model]
        batch, seq_len, _ = x.shape
        expanded = self.expand(x)  # [batch, seq_len, d_model * n_streams]
        return expanded.view(batch, seq_len, self.n_streams, self.d_model)

    def contract_from_streams(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Contrae múltiples streams a salida única."""
        # x_streams: [batch, seq_len, n_streams, d_model]
        batch, seq_len, _, _ = x_streams.shape
        flattened = x_streams.view(batch, seq_len, -1)  # [batch, seq_len, n_streams * d_model]
        return self.contract(flattened)  # [batch, seq_len, d_model]

    def forward(
        self,
        x: torch.Tensor,
        layer_output: torch.Tensor,
        is_first_layer: bool = False
    ) -> torch.Tensor:
        """
        Aplicar mHC residual.

        Args:
            x: Entrada/residual previo [batch, seq_len, d_model] o
               [batch, seq_len, n_streams, d_model] si no es primera capa
            layer_output: Salida de la capa (attention + FF) [batch, seq_len, d_model]
            is_first_layer: Si es la primera capa, expandir de 1 a n streams

        Returns:
            Streams actualizados [batch, seq_len, n_streams, d_model]
        """
        # Proyectar matrices al politopo de Birkhoff
        H_res = self.sinkhorn(self.H_res_raw)
        H_post = self.sinkhorn(self.H_post_raw)

        # Expandir si es primera capa
        if is_first_layer:
            x_streams = self.expand_to_streams(x)  # [batch, seq, n_streams, d_model]
        else:
            x_streams = x  # Ya está en formato de streams

        # Aplicar hyper-connection con restricción de manifold
        # x_{l+1} = H^{res} · x_l + H^{post,T} · F(x_l)

        batch, seq_len, n_streams, d_model = x_streams.shape

        # Mezcla residual: H_res @ x_streams
        # [n, n] @ [batch, seq, n, d] -> [batch, seq, n, d]
        x_res = torch.einsum('ij,bsjd->bsid', H_res, x_streams)

        # Mezcla de salida de capa: H_post.T @ layer_output
        # layer_output: [batch, seq, d] -> expandir a streams
        layer_expanded = layer_output.unsqueeze(2).expand(-1, -1, n_streams, -1)
        x_layer = torch.einsum('ji,bsjd->bsid', H_post, layer_expanded)

        # Combinación final
        x_new = x_res + x_layer

        return x_new

    def get_single_output(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Obtiene salida única desde streams (para última capa)."""
        return self.contract_from_streams(x_streams)


class mHCTransformerBlock(nn.Module):
    """
    Bloque Transformer con mHC residual.
    Reemplaza conexiones residuales estándar con Hyper-Connections
    proyectadas al politopo de Birkhoff.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int = 64,
        dim_ff: int = None,
        n_streams: int = 4,
        dropout: float = 0.1,
        sinkhorn_iterations: int = 5
    ):
        super().__init__()

        self.d_model = d_model
        self.n_streams = n_streams
        dim_ff = dim_ff or d_model * 4

        # Attention
        self.norm1 = nn.RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dim_head, dropout)

        # Feed-forward
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

        # mHC residuals
        self.mhc_attn = mHCResidual(d_model, n_streams, sinkhorn_iterations)
        self.mhc_ff = mHCResidual(d_model, n_streams, sinkhorn_iterations)

    def forward(
        self,
        x: torch.Tensor,
        is_first_layer: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass con mHC.

        Args:
            x: Input [batch, seq, d_model] si primera capa,
               [batch, seq, n_streams, d_model] si no
            is_first_layer: Flag para expansión inicial
            attention_mask: Máscara de atención opcional

        Returns:
            Streams actualizados [batch, seq, n_streams, d_model]
        """
        # Obtener representación única para pasar por attention
        if is_first_layer:
            x_single = x  # Ya es [batch, seq, d_model]
        else:
            # Promediar streams para attention
            x_single = x.mean(dim=2)  # [batch, seq, d_model]

        # Attention
        attn_out = self.attention(self.norm1(x_single), attention_mask)
        x = self.mhc_attn(x, attn_out, is_first_layer)

        # Feed-forward
        x_single = x.mean(dim=2)
        ff_out = self.ff(self.norm2(x_single))
        x = self.mhc_ff(x, ff_out, is_first_layer=False)

        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (reutilizada de main_network.py)."""

    def __init__(self, d_model: int, num_heads: int, dim_head: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(d_model, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch, seq_len, self.num_heads, self.dim_head).transpose(1, 2), qkv)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)
        return self.to_out(out)
```

### 3.2 Integración en H-Net Main Network

```python
# models/hnet/main_network_mhc.py

from .mhc_layers import mHCTransformerBlock, mHCResidual

class MainNetworkMHC(nn.Module):
    """
    H-Net Main Network con mHC Hyper-Connections.
    22 capas Transformer con streams residuales expandidos y
    matrices de mezcla proyectadas al politopo de Birkhoff.
    """

    def __init__(
        self,
        d_model: int = 1536,
        num_layers: int = 22,
        num_heads: int = 24,
        dim_head: int = 64,
        dim_ff: int = 6144,
        n_streams: int = 4,  # mHC streams
        dropout: float = 0.1,
        sinkhorn_iterations: int = 5
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.n_streams = n_streams

        # Capas mHC Transformer
        self.layers = nn.ModuleList([
            mHCTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_head=dim_head,
                dim_ff=dim_ff,
                n_streams=n_streams,
                dropout=dropout,
                sinkhorn_iterations=sinkhorn_iterations
            )
            for _ in range(num_layers)
        ])

        # Proyección final de streams a salida única
        self.final_contract = nn.Linear(d_model * n_streams, d_model)
        self.final_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input chunks [batch, seq_len, d_model]
            attention_mask: Optional mask

        Returns:
            Processed chunks [batch, seq_len, d_model]
        """
        # Primera capa: expandir a streams
        x = self.layers[0](x, is_first_layer=True, attention_mask=attention_mask)

        # Capas intermedias
        for layer in self.layers[1:]:
            x = layer(x, is_first_layer=False, attention_mask=attention_mask)

        # Contraer streams a salida única
        batch, seq_len, n_streams, d_model = x.shape
        x = x.view(batch, seq_len, -1)  # [batch, seq, n_streams * d_model]
        x = self.final_contract(x)  # [batch, seq, d_model]
        x = self.final_norm(x)

        return x
```

---

## 4. Configuración Recomendada

### 4.1 Config YAML: `configs/mhc_config.yaml`

```yaml
mhc:
  # Hyper-Connection streams
  n_streams: 4  # DeepSeek default

  # Sinkhorn-Knopp projection
  sinkhorn_iterations: 5  # Paper default
  sinkhorn_eps: 1.0e-8

  # Dónde aplicar mHC
  apply_to:
    hnet_main_network: true  # RECOMENDADO (22 capas)
    hrm_h_module: false       # Opcional (4 capas)
    hrm_l_module: false       # Opcional (4 capas)

  # Training stability
  warmup_mhc: true  # Warmup gradual de matrices H
  warmup_steps: 1000

  # Regularización
  orthogonality_reg: 0.01  # Fomentar diversidad en streams
```

### 4.2 Training Config Update

```yaml
# Añadir a configs/training_config.yaml

training:
  # ... existing config ...

  # mHC específico
  use_mhc: true
  mhc_config: "configs/mhc_config.yaml"

  # Ajustar learning rate (mHC es más estable)
  learning_rate: 2.0e-4  # Puede ser más alto que sin mHC

  # El overhead es ~6.7%, ajustar expectations
  expected_overhead: 0.067
```

---

## 5. Plan de Implementación

### Fase 1: Core mHC Module (2-3 días)
1. Implementar `SinkhornKnopp` module
2. Implementar `mHCResidual` block
3. Tests unitarios para proyección Birkhoff

### Fase 2: H-Net Integration (2-3 días)
1. Crear `MainNetworkMHC` alternativo
2. Config para switch entre estándar/mHC
3. Validar gradientes y estabilidad

### Fase 3: Training Validation (3-5 días)
1. Entrenar con mHC en dataset de test
2. Comparar loss curves (sin spikes)
3. Medir overhead real vs 6.7% esperado

### Fase 4: Optional HRM Integration (2 días)
1. Aplicar mHC a H-module y L-module
2. Evaluar beneficio en convergencia jerárquica

### Fase 5: Benchmarking (2-3 días)
1. Comparar métricas downstream
2. Documentar mejoras
3. Actualizar README

---

## 6. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Overhead > 6.7% | Media | Bajo | Optimizar Sinkhorn iterations |
| Incompatibilidad Mamba-2 | Baja | Alto | mHC solo en Main Network Transformer |
| Memory overhead | Media | Medio | n_streams=4 máximo, gradient checkpointing |
| Convergencia lenta inicial | Media | Bajo | Warmup de matrices H |

---

## 7. Métricas de Éxito

### Training Stability
- [ ] Sin loss spikes hasta step 50k
- [ ] Gradient norm suave (sin explosiones)

### Performance
- [ ] Accuracy +5-7% en tasks downstream
- [ ] APM (Accuracy Per Million) mejorado

### Efficiency
- [ ] Overhead < 10% en tiempo de entrenamiento
- [ ] Latencia inferencia sin cambio significativo

---

## 8. Conclusión

**mHC es una adición de alto valor para CodKing** porque:

1. **Resuelve un problema real**: Las 22 capas del Main Network de H-Net pueden beneficiarse significativamente de mejor estabilidad de entrenamiento.

2. **Overhead mínimo**: Solo 6.7% de tiempo adicional de entrenamiento.

3. **Implementación modular**: Se puede añadir como módulo separado sin modificar arquitectura existente significativamente.

4. **Validado por DeepSeek**: La técnica se usa en modelos de producción de 3B-27B parámetros.

5. **Compatible con stack existente**: PyTorch nativo, sin dependencias adicionales.

**Recomendación**: Proceder con implementación de Fase 1-2 y evaluar beneficios antes de integración completa.

---

## Referencias

- [mHC Paper (arXiv:2512.24880)](https://arxiv.org/abs/2512.24880)
- [DeepSeek mHC Implementation](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)
- [Hugging Face Paper Page](https://huggingface.co/papers/2512.24880)
- [Hacker News Discussion](https://news.ycombinator.com/item?id=46452172)
- [Visual Explanation: The Manifold Dial](https://subhadipmitra.com/blog/2026/deepseek-mhc-manifold-constrained-hyper-connections/)

---

*Documento creado: 2026-01-09*
*Última actualización: 2026-01-09*
*Autor: Claude Code Analysis*
