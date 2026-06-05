
1- تثبيت المكتبات:
pip install -r requirements.txt

2- تدريب النماذج:
python train.py

3- تشغيل API:
uvicorn api.main:app --reload

4- فتح Swagger:
http://127.0.0.1:8000/docs
