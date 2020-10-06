FROM python:3.6
WORKDIR /visualsax

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

ENV LANG C.UTF-8

CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf.py"]
