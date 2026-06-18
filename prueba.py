"""
multimodal_dataset.py  —  VERSIÓN COMENTADA
============================================
Este módulo es el PUENTE entre tus datos en disco (un CSV + una carpeta de
imágenes) y el autoencoder. Su única responsabilidad: leer cada muestra,
prepararla, y entregarla en el formato exacto que el modelo espera
(un diccionario de 4 claves: numeric, multiclass, binary, image).

Tiene tres piezas:
  1. Preprocessor         -> aprende cómo transformar los datos tabulares
  2. MultimodalCSVDataset -> une una fila del CSV con su imagen, por muestra
  3. Schema / build_schema -> resume las dimensiones para construir el modelo
"""

# --- IMPORTS: qué aporta cada librería ---
import os                              # construir rutas de archivo (carpeta + nombre img)
from dataclasses import dataclass      # crear la clase Schema sin escribir __init__ a mano
from typing import List, Dict          # anotaciones de tipo (solo documentan, no obligan)

import numpy as np                     # cálculo numérico: medias, arrays, estandarización
import pandas as pd                    # leer y manipular el CSV como tabla (DataFrame)
import torch                           # convertir los arrays en tensores para el modelo
from torch.utils.data import Dataset   # clase base que PyTorch sabe recorrer con DataLoader
from PIL import Image                  # abrir y manipular archivos de imagen (PNG/JPG)


# ============================================================================
# PIEZA 1 — PREPROCESADOR
# Su trabajo: APRENDER del train cómo transformar los datos tabulares, y luego
# APLICAR esa misma transformación a cualquier split. Separar "aprender" de
# "aplicar" es lo que evita la fuga de información (data leakage).
# ============================================================================
class Preprocessor:
    """
    Recibe QUÉ columna es de cada tipo. No transforma nada todavía;
    eso ocurre en fit() (aprender) y transform_row() (aplicar).
    """
    def __init__(self, numeric_cols: List[str],
                 multiclass_cols: List[str],
                 binary_cols: List[str]):
        # Guarda los nombres de columna de cada tipo. list(...) hace una copia
        # defensiva para que, si el usuario modifica su lista después, no afecte aquí.
        self.numeric_cols = list(numeric_cols)
        self.multiclass_cols = list(multiclass_cols)
        self.binary_cols = list(binary_cols)

        # --- Parámetros que se RELLENAN en fit(). Arrancan vacíos (None / dict vacío) ---
        # Para estandarizar numéricos necesitamos su media y desviación.
        self.num_mean: np.ndarray | None = None
        self.num_std: np.ndarray | None = None
        # Para cada columna multiclase: un diccionario {valor_original -> índice entero}.
        # Ej: {"azul":0, "rojo":1, "verde":2}. Es el "vocabulario" de esa columna.
        self.mc_vocab: Dict[str, Dict] = {}
        # Para cada columna binaria: {valor_original -> 0.0 ó 1.0}. Ej: {"no":0.0, "si":1.0}.
        self.bin_map: Dict[str, Dict] = {}

    def fit(self, df: pd.DataFrame):
        """APRENDE las transformaciones. Llamar SOLO con el DataFrame de train."""
        # --- NUMÉRICOS: calcular media y desviación de cada columna ---
        if self.numeric_cols:                                  # si hay columnas numéricas
            arr = df[self.numeric_cols].to_numpy(dtype=np.float32)  # tabla -> array (N, n_num)
            self.num_mean = arr.mean(axis=0)                   # media por columna
            self.num_std = arr.std(axis=0)                     # desviación por columna
            # Guardia: si una columna es constante, su std=0 y dividir daría infinito/NaN.
            # La forzamos a 1 para que esa columna simplemente quede centrada en 0.
            self.num_std[self.num_std == 0] = 1.0

        # --- MULTICLASE: construir el vocabulario de cada columna ---
        for col in self.multiclass_cols:
            # .dropna() ignora vacíos; .unique() saca los valores distintos.
            # Se ordenan (key=str) para que el mapeo sea DETERMINISTA entre ejecuciones:
            # "azul" siempre será 0, etc. Sin ordenar, el orden podría variar.
            cats = sorted(df[col].dropna().unique().tolist(), key=lambda v: str(v))
            # Asignar un índice entero consecutivo a cada categoría.
            self.mc_vocab[col] = {val: i for i, val in enumerate(cats)}

        # --- BINARIOS: mapear sus dos valores a 0.0 y 1.0 ---
        for col in self.binary_cols:
            vals = sorted(df[col].dropna().unique().tolist(), key=lambda v: str(v))
            # Una columna "binaria" con más de 2 valores es un error de configuración:
            # avisamos en vez de continuar silenciosamente.
            if len(vals) > 2:
                raise ValueError(f"La columna binaria '{col}' tiene >2 valores: {vals}")
            # Funciona tanto si ya son 0/1 como si son texto ("no"/"si"): el primero
            # alfabéticamente -> 0.0, el segundo -> 1.0.
            self.bin_map[col] = {v: float(i) for i, v in enumerate(vals)}
        return self   # devolver self permite encadenar: Preprocessor(...).fit(df)

    def transform_row(self, row: pd.Series):
        """APLICA lo aprendido a UNA fila. Devuelve 3 arrays: numéricos, multiclase, binarios."""
        # --- NUMÉRICOS: (valor - media) / desviación  =>  centrados en 0, escala 1 ---
        if self.numeric_cols:
            num = np.array([row[c] for c in self.numeric_cols], dtype=np.float32)
            num = (num - self.num_mean) / self.num_std         # estandarización
        else:
            num = np.zeros(0, dtype=np.float32)                # array vacío si no hay numéricos

        # --- MULTICLASE: cada valor -> su índice del vocabulario ---
        # .get(valor, 0): si la categoría NO se vio en train (aparece solo en val/test),
        # cae a 0 en vez de reventar. Es una decisión de seguridad; vigílala si tienes
        # categorías raras. dtype int64 porque nn.Embedding exige índices enteros.
        mc = np.array(
            [self.mc_vocab[c].get(row[c], 0) for c in self.multiclass_cols],
            dtype=np.int64,
        )

        # --- BINARIOS: cada valor -> 0.0/1.0 según el mapeo ---
        bn = np.array(
            [self.bin_map[c].get(row[c], 0.0) for c in self.binary_cols],
            dtype=np.float32,
        )
        return num, mc, bn

    @property
    def multiclass_cardinalities(self) -> List[int]:
        """Cuántas categorías tiene cada columna multiclase. El modelo necesita esto
        para dimensionar los Embeddings y los heads de salida. @property permite
        leerlo como atributo: pre.multiclass_cardinalities (sin paréntesis)."""
        return [len(self.mc_vocab[c]) for c in self.multiclass_cols]


# ============================================================================
# PIEZA 2 — DATASET
# Su trabajo: para un índice i, devolver la muestra i COMPLETA (tabular + imagen)
# en el formato del modelo. PyTorch lo recorre con un DataLoader, que se encarga
# de agrupar muestras en batches y barajarlas.
# ============================================================================
class MultimodalCSVDataset(Dataset):       # hereda de Dataset -> compatible con DataLoader
    def __init__(self, df: pd.DataFrame, pre: Preprocessor,
                 image_dir: str, image_col: str,
                 img_size: int = 32, img_channels: int = 3):
        # reset_index: tras un split las filas conservan índices originales (ej. 5,9,12...).
        # Lo reseteamos a 0,1,2... para poder acceder por posición con .iloc[idx].
        self.df = df.reset_index(drop=True)
        self.pre = pre                      # el MISMO preprocesador ya ajustado con train
        self.image_dir = image_dir          # carpeta donde están las imágenes
        self.image_col = image_col          # nombre de la columna que tiene el nombre de archivo
        self.img_size = img_size
        self.img_channels = img_channels    # 3 = RGB, 1 = escala de grises

    def __len__(self):
        """Cuántas muestras hay. El DataLoader lo usa para saber cuándo terminar."""
        return len(self.df)

    def _load_image(self, filename: str) -> torch.Tensor:
        """Carga UNA imagen de disco y la convierte en tensor listo para el modelo.
        El guion bajo inicial (_load_image) marca que es un método interno de ayuda."""
        path = os.path.join(self.image_dir, str(filename))    # carpeta + nombre -> ruta completa
        mode = "RGB" if self.img_channels == 3 else "L"       # "L" = grises en PIL
        with Image.open(path) as im:                          # 'with' cierra el archivo al salir
            im = im.convert(mode)                             # asegura el nº de canales correcto
            # Seguridad: aunque dijiste que ya son uniformes, forzamos el tamaño esperado.
            if im.size != (self.img_size, self.img_size):
                im = im.resize((self.img_size, self.img_size))
            # A array y NORMALIZAR a [0,1] dividiendo entre 255 (los píxeles vienen en 0..255).
            # El modelo espera imágenes en [0,1] porque el decoder termina en Sigmoid.
            arr = np.asarray(im, dtype=np.float32) / 255.0
        # PyTorch espera el orden (Canales, Alto, Ancho). PIL/numpy dan (Alto, Ancho, Canales).
        if self.img_channels == 1:
            arr = arr[None, :, :]                # grises: añade dim de canal -> (1, H, W)
        else:
            arr = np.transpose(arr, (2, 0, 1))   # color: (H,W,C) -> (C,H,W)
        return torch.from_numpy(arr)             # array numpy -> tensor torch

    def __getitem__(self, idx: int):
        """EL CORAZÓN: dado un índice, devuelve la muestra completa como diccionario.
        El DataLoader llama a esto repetidamente y apila los resultados en batches."""
        row = self.df.iloc[idx]                          # la fila i del CSV
        num, mc, bn = self.pre.transform_row(row)        # parte tabular ya transformada
        image = self._load_image(row[self.image_col])    # parte de imagen
        # Las claves DEBEN coincidir con las que el modelo lee en encode().
        return {
            "numeric":    torch.from_numpy(num),     # (n_numeric,) float32
            "multiclass": torch.from_numpy(mc),      # (n_mc_cols,) int64  <- índices
            "binary":     torch.from_numpy(bn),      # (n_binary,)  float32
            "image":      image,                     # (C, H, W)    float32 en [0,1]
        }


# ============================================================================
# PIEZA 3 — SCHEMA
# Su trabajo: empaquetar las dimensiones del dataset (cuántas columnas de cada
# tipo, cardinalidades, tamaño de imagen) para construir el modelo con los
# números correctos SIN tener que contarlos a mano.
# ============================================================================
@dataclass                              # genera __init__, __repr__, etc. automáticamente
class Schema:
    n_numeric: int                      # nº de columnas numéricas
    multiclass_cardinalities: List[int] # nº de categorías de cada columna multiclase
    n_binary: int                       # nº de columnas binarias
    img_channels: int                   # 3 (RGB) o 1 (grises)
    img_size: int                       # lado de la imagen (asume cuadrada)


def build_schema(pre: Preprocessor, img_channels: int, img_size: int) -> Schema:
    """Lee las dimensiones desde el preprocesador (ya ajustado) y arma el Schema.
    Lo de las imágenes se pasa aparte porque el preprocesador solo conoce lo tabular."""
    return Schema(
        n_numeric=len(pre.numeric_cols),
        multiclass_cardinalities=pre.multiclass_cardinalities,
        n_binary=len(pre.binary_cols),
        img_channels=img_channels,
        img_size=img_size,
    )