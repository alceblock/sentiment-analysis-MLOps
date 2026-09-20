# sentiment-analysis-MLOps

This repository implements a production-grade, containerized MLOps architecture designed for automated sentiment analysis and continuous online reputation tracking. While standard text classification models often remain isolated in local environments, this project engineers a robust operational framework that bridges the gap between deep learning and scalable software deployment.

The system is architected as an end-to-end containerized microservices infrastructure:
* **Production API & Containerization:** Enclosing the `twitter-roberta-base-sentiment-latest` transformer model within a high-performance FastAPI backend, fully containerized using Docker and orchestrated via Docker Compose for rapid scalability and local execution.
* **Continuous Monitoring & Analytics:** Building an live operational telemetry layer by connecting the prediction engine to a Prometheus data aggregator and visualizing brand perception dynamics through Grafana dashboards (tracking metrics like accuracy, F1-score, and positive/negative text counters).
* **Automated CI/CD Delivery:** Designing a complete GitHub Actions automation pipeline that triggers automated test suites (`pytest`), handles model retraining validation, and manages continuous deployment directly to the Hugging Face Spaces ecosystem.

The final project delivers a resilient, cloud-ready monitoring system, demonstrating practical software engineering principles applied to real-time AI lifecycle management.

To see more, an extended technical explanation can be found in the project’s README.md file.

> [!WARNING]
> Project Root: All source code and MLOps pipelines are located in the MLOps_sentiment_project/ directory to keep separate high-level documentation and code. To execute the project, treat MLOps_sentiment_project/ as root accordingly.
> 
> * **CI/CD Configuration:** If you are running GitHub Actions, remember to update your workflow files (e.g., `.github/workflows/CI_CD.yml`) by `adding defaults:      run: working-directory: ./MLOps_sentiment_project`.
> * **Standalone Version (Suggested):** Alternatively, you can check out the [standalone MLOps_sentiment_project version](https://github.com/alceblock/MLOps_sentiment_project) where this project runs directly from the root.
