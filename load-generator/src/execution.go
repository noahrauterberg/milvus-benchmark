package main

import (
	"context"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const arrivalSeed = 3456

func ExecuteBenchmark(
	c *milvusclient.Client,
	collection string,
	vecFieldName string,
	datasource DataSource,
	dim int,
	jobGenParams JobGenerationParameters,
	k int,
	concurrency int,
) ([]Job, []UserSession, error) {
	ctx := context.Background()
	logger, err := NewLogger("benchmark")
	if err != nil {
		return nil, nil, err
	}
	defer logger.Close()
	logger.Log("Executing Benchmark...")

	/* Load Collection */
	task, err := c.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collection))
	if err != nil {
		return nil, nil, err
	}
	task.Await(ctx)

	/* Create Arrival Controller for Poisson-Process based workload */
	arrivalController := NewArrivalController(
		jobGenParams,
		dim,
		arrivalSeed,
		concurrency,
	)

	logger.Logf("Starting Benchmark with Poisson arrivals: targetQPS=%.2f, duration=%v, jobProbability=%.2f",
		jobGenParams.targetQPS, jobGenParams.benchmarkDuration, jobGenParams.jobProbability)

	/* Execute Workload with Poisson arrivals */
	jobs, sessions := ExecuteWorkloadPoisson(
		arrivalController,
		c,
		collection,
		vecFieldName,
		dim,
		k,
		logger,
		concurrency,
	)
	logger.Log("Finished Execution")

	return jobs, sessions, nil
}
