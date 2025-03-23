# price_prediction_api.py
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from main import get_db_connection  # main.py에서 get_db_connection 임포트

# CNN 모델 정의
class PricePredictionCNN(nn.Module):
    def __init__(self):
        super(PricePredictionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# FastAPI 앱 생성
app = FastAPI(title="Clothing Price Prediction API")

# 전역 변수로 모델과 전처리 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PricePredictionCNN().to(device)
model_path = "price_prediction_model.pth"

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 앱 시작 시 모델 로드 및 DB 연결 테스트
@app.on_event("startup")
async def startup_event():
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")

        # DB 연결 테스트
        conn = get_db_connection()
        conn.close()
        print("Database connection tested successfully")
    except Exception as e:
        print(f"Error during startup: {str(e)}")

# 가격 예측 함수
def predict_price(image_bytes: bytes) -> float:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image)
        return prediction.item()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# API 엔드포인트
@app.post("/predict_price/")
async def predict_price_endpoint(file: UploadFile):
    """
    Upload an image and get the predicted price
    Returns: JSON with predicted price
    """
    try:
        # 파일 읽기
        image_bytes = await file.read()

        # 가격 예측
        predicted_price = predict_price(image_bytes)

        return JSONResponse(
            content={
                "filename": file.filename,
                "predicted_price": round(predicted_price, 2),
                "currency": "USD"
            },
            status_code=200
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# 건강 체크 엔드포인트 (DB 연결 상태 포함)
@app.get("/health")
async def health_check():
    db_status = "healthy"
    try:
        conn = get_db_connection()
        conn.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "db_status": db_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)