# AWS deployment notes (design only, nothing deployed)

These are notes on how this service *would* run on AWS. No cloud resources were
created for this project and nothing here has been tested on AWS.

## Shape

```
            +-------------------+        +-------------------------+
 client --> | API Gateway / ALB | -----> | ECS Fargate: cvlab API  |  /classify /search /copy
            +-------------------+        |  (CPU, stub-free image)  |
                                         +-----------+-------------+
                                                     | SQS (generate jobs)
                                                     v
                                         +-------------------------+
                                         | ECS on EC2 g5/g6 (GPU)  |  /generate workers:
                                         |  rembg + diffusion +    |  segment -> scene ->
                                         |  quality gate           |  gate
                                         +-----+-------------+-----+
                                               |             |
                                       pass    v             v  fail
                                    S3 assets/approved/   S3 assets/review/ + DynamoDB
                                                          review queue -> reviewer UI
```

## Components

| Concern | Choice | Why |
|---|---|---|
| Product images and generated assets | S3 (`raw/`, `approved/`, `review/` prefixes), SSE-KMS, versioning, lifecycle rule expiring `review/` rejects | Generated images are cheap to recreate; originals are not |
| Embeddings / search index | Start with FAISS loaded in the API task from an S3 snapshot (rebuilt nightly). Move to **pgvector on Aurora PostgreSQL** or **OpenSearch k-NN** once the catalog needs live updates or filtered search | FAISS flat index on ~100k x 512 floats is ~200 MB and exact; the managed options add filtering, updates and HA |
| Classification + search API | ECS Fargate, CPU, 2+ tasks behind an ALB. CLIP ViT-B/32 image embedding on CPU is fine for interactive use; batch catalog embedding runs as a separate job | No GPU needed for request-time text embedding + index lookup |
| Generation | Async: API puts a job on **SQS**, GPU workers on **ECS on EC2** (g5/g6) or **SageMaker async inference** consume it, write to S3, and notify via EventBridge | Diffusion is seconds per image and bursty; never on the request thread |
| Review queue | DynamoDB table (`review_id` PK, `status` GSI) instead of the local JSONL file | Same API as `ReviewQueue`; conditional writes make decisions idempotent |
| LLM copy | Claude via the Anthropic API or Amazon Bedrock; API keys in Secrets Manager. Guardrails run in-process exactly as in `cvlab.copywriting` | The guardrail layer is provider-agnostic |
| Lambda? | Good for the thin parts (S3 upload trigger -> enqueue embedding job, review-decision webhook). Not for model serving: torch + CLIP exceeds comfortable Lambda package/cold-start limits | |
| Observability | CloudWatch metrics for gate pass rate, review-queue depth, fallback-to-template rate, p95 latency; alarm when gate pass rate drops | These are the production versions of the numbers in `results/` |

## Things to settle before a real deploy

- Model licences: SD-Turbo ships under the Stability AI Community License (its model
  card points commercial users to stability.ai/license; the licence has revenue-based
  terms). Check those terms, or use a hosted image API, before any commercial use.
- Identity: img2img redraws faces (see README results). On-model imagery of real
  people needs consent and must not be altered without it.
- Data: the dataset here is a public Kaggle set used for learning only.
