FROM python:3.11-slim

WORKDIR /app

# 构建上下文应为项目根目录（含 ecom_agent_matrix/）
COPY ecom_agent_matrix/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir uvicorn[standard]

COPY ecom_agent_matrix /app/ecom_agent_matrix

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "ecom_agent_matrix.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
