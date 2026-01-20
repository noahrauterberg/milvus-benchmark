# Milvus Vector Database Benchmarking Project

This project was created as part of the "Cloud Service Benchmarking" course at the TU Berlin and implements a benchmark for the Milvus Vector Database, focussing on the HNSW index implementation in particular.
The benchmark evaluates mainly the index build time, query latency, and recall (i.e. approximate neighbor accuracy) using a synthetic workload on different dimensionalities of the GloVe dataset.

## Project Structure

This repository hosts the cloud infrastructure definition as well as the implementation of the load generator.
For more detailed information, please refer to the documentation in the respective directory:
`terraform` - configuration of the gcp infrastructure
`load-generator` - implementation of the load generator

## Benchmark Design

The benchmark aims to evaluate the following qualities and characterize the tradeoff between them:
* **Performance**, measured by the dimensions of query latency and index construction time
* **Response accuracy**, measured by recall (the fraction of true nearest neighbors returned by the approximate search)

### Dataset

The benchmark uses the **GloVe (Global Vectors for Word Representation)** dataset, which contains pre-trained word embeddings.
These embeddings are real-world vector representations that capture semantic relationships between words, making them highly relevant for evaluating vector similarity search—a common use case for vector databases in NLP applications like semantic search, recommendation systems, and RAG pipelines.

The dataset is available in multiple dimensionalities (50, 100, 200, 300), allowing the benchmark to evaluate how vector dimension affects performance and accuracy.

Note that the dataset may easily be replaced with any other vector dataset of similar structure, as the benchmark is designed to be dataset-agnostic.

### Synthetic Workload

The benchmark generates a synthetic workload consisting of two types of work units:
* _Simple Jobs_: Independent k-NN queries using randomly generated vectors (sampled from a normal distribution with configurable mean and standard deviation)
* _Simulated User Sessions_: Sequential, dependent queries that model realistic user behavior. Each session starts with a random query, and subsequent queries are derived from the top result of the previous query plus a small random offset—simulating attention-based drift as a user explores similar items.

This mixed workload reflects real-world usage patterns where some queries are independent while others form coherent search sessions.

### Partly-Open Arrival Model

The benchmark implements a partly-open arrival model:
* A fixed number of concurrent workers (configurable via `concurrency`) continuously pull work from a shuffled queue of jobs and sessions
* Workers execute work units as fast as possible without artificial delays between requests
* User sessions are treated as atomic units—all queries within a session are executed sequentially by a single worker before the worker picks up new work

This model balances the realism of session-based workloads with the stress-testing capability of a closed-loop system, allowing the benchmark to measure maximum sustainable throughput while still capturing session-level latency characteristics.

## Architecture

The benchmark is designed to run in a cloud environment, utilizing two VMs.

The SUT is deployed as a Milvus Standalone instance in a gcp `e2-standard-4` instance, following the recommended resource specification.

The Load Generator is deployed on a separate VM, using the Milvus go-sdk to communicate with the SUT.

## Implementation Overview

As every good benchmark, the benchmark is executed in four phases:

### Preparation

The benchmark starts with a short preparation phase, in which the GloVe dataset is inserted in batches, random query vectors are generated, and the HNSW index is created.

### Warmup

To ensure realistic behavior and stabilize the SUT, the benchmark starts off with a few warmup requests with random queries.
Note that the neither the queries nor the responses are logged and therefore not considered in the analysis.

## Execution

When the SUT is in a stable state, the actual benchmark is executed using the previously generated random vectors.
Since the benchmark implements a partly open arrival model, two types of workload are used:
* _Simple Job_: Independent queries
* _Simulated User Session_: Sequential queries, executed one after another where each follow-up query is based on the previous top result, offset by a small random vector to simulate an attention-based change to the previous output.

## Collection/Cleanup

After all queries have been executed, the response accuracy is calculated by calculating the exact nearest neighbors for each query vector.
Note that for user sessions, this can only be done after all queries have completed since the query vectors are not know before the previous query has been answered.
Finally, the results are written to a parquet file for later analysis.

After downloading the result and log files, the infrastructure may be shut down.

