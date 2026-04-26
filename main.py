from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# PyTorch model sınıfı – 5 giriş özelliği ile çalışır.
class BreastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.linear_layer_stack(x)


# İstek gövdesini doğrulamak için Pydantic modeli.
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=5, max_length=5)


model: BreastCancerModel | None = None
scaler = None

# Her ozellik icin: ingilizce ad, turkce ad, min, max
FEATURE_INFO = [
    {"name": "Worst Perimeter", "name_tr": "En Kötü Çevre Uzunluğu", "min": 50.41, "max": 251.20},
    {"name": "Worst Area", "name_tr": "En Kötü Alan", "min": 185.20, "max": 4254.00},
    {"name": "Worst Concave Points", "name_tr": "En Kötü İçbükey Noktalar", "min": 0.0, "max": 0.291},
    {"name": "Mean Concave Points", "name_tr": "Ortalama İçbükey Noktalar", "min": 0.0, "max": 0.201},
    {"name": "Worst Radius", "name_tr": "En Kötü Yarıçap", "min": 7.93, "max": 36.04},
]

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, scaler

    # Uygulama ayağa kalkarken model ve scaler belleğe alınır.
    model_path = BASE_DIR / "models" / "breast_cancer_model_top5.pth"
    scaler_path = BASE_DIR / "models" / "cancer_scaler_top5.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler dosyasi bulunamadi: {scaler_path}")

    loaded_model = BreastCancerModel()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()  # Inference icin eval moduna alinir.

    loaded_scaler = joblib.load(scaler_path)

    model = loaded_model
    scaler = loaded_scaler
    yield


app = FastAPI(title="Meme Kanseri Teshis API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def run_prediction(features: List[float]) -> dict:
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model veya scaler yuklenemedi.")

    # Gelen 5 ozellik degerini scaler ile olceklendiriyoruz.
    scaled_features = scaler.transform([features])

    # Numpy ciktiyi tensor'e cevirip model tahmini yapiyoruz.
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = int(torch.argmax(output, dim=1).item())
        confidence = probabilities[0][prediction].item()

    # Sonucu istenen etiketlere donusturuyoruz.
    if prediction == 1:
        result_text = "İyi Huylu (Benign)"
        risk_level = "low"
    else:
        result_text = "Kötü Huylu (Malignant)"
        risk_level = "high"

    return {
        "prediction": prediction,
        "result": result_text,
        "confidence": round(confidence * 100, 1),
        "risk_level": risk_level,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Ana sayfada kullanicidan 5 deger alip API'a gonderen arayuz sunulur.
    return templates.TemplateResponse(
        request, "index.html", {"features": FEATURE_INFO}
    )


@app.get("/health")
def health_check():
    # API durumunu kontrol etmek icin ayri endpoint.
    return {"status": "ok", "message": "API calisiyor."}


@app.post("/predict")
def predict(payload: PredictionRequest):
    return run_prediction(payload.features)
