FROM python:3.12-alpine
WORKDIR /app
COPY README.md index.html app.js ./
RUN adduser -D -u 10001 lab
USER lab
EXPOSE 7860
CMD ["python", "-m", "http.server", "7860", "--bind", "0.0.0.0"]
