"""
cv_data.py — Replace this content with YOUR actual CV/resume.
This is the only file you need to edit to personalize the bot.
"""

CV_TEXT = """
ALEX MORGAN
MLOps Engineer | AI/ML Platform Specialist
Email: alex.morgan@email.com | LinkedIn: linkedin.com/in/alexmorgan | GitHub: github.com/alexmorgan
Location: San Francisco, CA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROFESSIONAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MLOps Engineer with 5+ years of experience designing and deploying scalable ML infrastructure. 
Specialized in building production-grade RAG pipelines, LLM serving systems, and end-to-end 
ML platforms. Passionate about bridging the gap between research prototypes and production systems.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORK EXPERIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Senior MLOps Engineer — TechCorp AI (2022–Present)
• Engineered a production-ready RAG pipeline using Hugging Face embeddings and FAISS, 
  implementing semantic search over unstructured documents to reduce hallucination in 
  LLM responses by 40%.
• Designed and deployed an LLM serving infrastructure on Kubernetes using vLLM and 
  TGI (Text Generation Inference), achieving 3x throughput improvement over naive serving.
• Built a real-time ML feature store using Redis and Apache Kafka, reducing feature 
  computation latency from 500ms to under 20ms for 15 production models.
• Led migration of monolithic ML training pipelines to Kubeflow Pipelines, cutting 
  experiment cycle time by 60% and improving reproducibility across 8-person team.
• Implemented model monitoring with Evidently AI and custom drift detection, preventing 
  3 silent model degradation incidents that would have impacted 50K+ daily users.

ML Platform Engineer — DataDriven Inc. (2020–2022)
• Architected a multi-tenant ML platform serving 200+ data scientists, built on 
  MLflow, DVC, and AWS SageMaker.
• Reduced model deployment time from 2 weeks to 4 hours by building a self-serve 
  CI/CD pipeline using GitHub Actions and ArgoCD.
• Optimized transformer model inference using ONNX Runtime and quantization techniques, 
  achieving 4x speedup with less than 1% accuracy degradation.
• Built an automated data labeling pipeline integrating Label Studio with active learning, 
  reducing annotation costs by 35%.

Data Engineer — StartupXYZ (2019–2020)
• Designed and maintained Apache Spark ETL pipelines processing 10TB+ daily data.
• Built real-time dashboards using Apache Flink and Apache Superset for business KPIs.
• Implemented data quality checks using Great Expectations, catching 95% of upstream 
  data issues before model training.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MLOps & Infrastructure:
  Kubernetes, Docker, Kubeflow, MLflow, DVC, Weights & Biases, ArgoCD, Terraform, 
  AWS SageMaker, Google Vertex AI, Azure ML

LLM & AI:
  RAG pipelines, FAISS, ChromaDB, LangChain, Hugging Face Transformers, TGI, vLLM,
  LoRA fine-tuning, ONNX, TensorRT, Prompt Engineering, LLM Evaluation

Languages & Frameworks:
  Python (expert), Bash, SQL, Go (intermediate), PyTorch, TensorFlow, FastAPI, 
  Gradio, Streamlit

Data Engineering:
  Apache Spark, Apache Kafka, Apache Flink, dbt, Airflow, Prefect, Redis, 
  PostgreSQL, Elasticsearch

Monitoring & Observability:
  Prometheus, Grafana, Evidently AI, WhyLabs, Datadog, OpenTelemetry

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EDUCATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

M.S. Computer Science — Stanford University (2019)
  Specialization: Machine Learning Systems
  Thesis: "Efficient Serving of Large Language Models at Scale"

B.S. Computer Science — UC Berkeley (2017)
  Minor: Statistics
  GPA: 3.8/4.0 | Dean's List (4 semesters)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OpenRAG — Open-source RAG Framework (2023)
  Built a production-ready RAG framework with 1.2K GitHub stars. Features include 
  hybrid search (dense + sparse), re-ranking, query decomposition, and streaming responses.
  Tech: Python, FAISS, sentence-transformers, FastAPI, Redis
  GitHub: github.com/alexmorgan/openrag

LLM Cost Optimizer (2023)
  Tool that automatically routes LLM requests to the cheapest model capable of handling 
  the task, reducing API costs by 70% in production.
  Tech: Python, Claude API, OpenAI API, FastAPI

MLOps Template (2022)
  Production-grade MLOps project template with CI/CD, monitoring, and automated testing 
  built-in. Used by 500+ ML engineers.
  Tech: Python, GitHub Actions, MLflow, Docker, Kubernetes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CERTIFICATIONS & SPEAKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Certifications:
  • AWS Certified Machine Learning Specialty (2023)
  • Certified Kubernetes Administrator (CKA) (2022)
  • Google Professional ML Engineer (2021)

Speaking:
  • MLOps World 2023: "From Notebook to Production: RAG Pipelines at Scale"
  • PyData Global 2022: "Lessons from Running 50 Models in Production"
  • KubeCon 2022: "ML Workloads on Kubernetes: Patterns and Anti-patterns"

Publications:
  • "Efficient RAG: Reducing Latency and Hallucination in Production LLM Systems" 
    — Towards Data Science (2023, 50K+ views)
  • "The Hidden Costs of ML in Production" — O'Reilly Radar (2022)
"""

# You can also add notes, blog posts, or any other documents here
ADDITIONAL_NOTES = """
Interview Preparation Notes — Alex Morgan

Why I want to work in ML/AI:
I've always been fascinated by systems that learn. My first exposure to ML was during 
my undergrad when I implemented a basic neural network from scratch. Seeing it learn 
to recognize handwritten digits felt like magic. Now I focus on making that magic 
reliable and scalable in production environments.

My biggest technical challenge:
At TechCorp, we had a RAG system that worked perfectly in testing but would occasionally 
return completely wrong answers in production. The root cause was distribution shift 
in how users phrased questions vs our test set. I built an online evaluation system 
that caught these cases using LLM-as-judge + user feedback signals, then implemented 
adaptive retrieval strategies. This reduced bad responses by 40%.

What I look for in a team:
Strong engineering culture, psychological safety to experiment and fail fast, 
and a focus on real impact over vanity metrics.

Salary expectations:
Targeting $200K-$250K base depending on the company stage, benefits, and equity.

Availability:
Can start in 2-4 weeks with notice.
"""