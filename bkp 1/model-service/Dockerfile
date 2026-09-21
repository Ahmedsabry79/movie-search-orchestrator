FROM vllm/vllm-openai:latest

WORKDIR /srv/model-service

COPY gateway.py /srv/model-service/gateway.py
COPY start.sh /srv/model-service/start.sh

RUN chmod +x /srv/model-service/start.sh

EXPOSE 8000
ENTRYPOINT ["/srv/model-service/start.sh"]
