package main

import (
	"context"
	"fmt"
	"math/rand"
	"sync"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func Warmup(
	c *milvusclient.Client,
	numberWarmupQueries int,
	dim int,
	collection string,
	vecFieldName string,
	k int,
) error {
	ctx := context.Background()
	logger, err := NewLogger("warmup")
	if err != nil {
		return err
	}
	defer logger.Close()
	logger.Log("Warming up...")

	/* Load Collection */
	task, err := c.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collection))
	if err != nil {
		return err
	}
	task.Await(ctx)

	/* Generate Random Warmup Queries */
	warmupJobs := generateWarmupJobs(
		rand.New(rand.NewSource(420)),
		dim,
		10.0,
		100.0,
		numberWarmupQueries,
	)

	/* Execute Warmup Queries - closed-loop, as fast as possible */
	executeWarmup(
		warmupJobs,
		c,
		collection,
		vecFieldName,
		k,
		logger,
		7, // number of workers
	)

	return nil
}

// generateWarmupJobs creates simple warmup jobs (without full Job struct overhead)
func generateWarmupJobs(
	generator *rand.Rand,
	dim int,
	stdDev float32,
	mean float32,
	numJobs int,
) []Vector {
	jobs := make([]Vector, numJobs)
	for i := range numJobs {
		jobs[i] = GenerateVector(generator, dim, stdDev, mean)
	}
	return jobs
}

// executeWarmup runs warmup queries as fast as possible (closed-loop, no timing)
func executeWarmup(
	queries []Vector,
	c *milvusclient.Client,
	collection string,
	vecFieldName string,
	k int,
	logger *Logger,
	numWorkers int,
) {
	workChan := make(chan Vector, numWorkers*2)

	var wg sync.WaitGroup
	for i := range numWorkers {
		wg.Add(1)
		go func(workerId int) {
			defer wg.Done()
			ctx := context.Background()
			for query := range workChan {
				_, err := c.Search(ctx,
					milvusclient.NewSearchOption(
						collection,
						k,
						[]entity.Vector{entity.FloatVector(query)},
					).WithANNSField(vecFieldName),
				)
				if err != nil {
					logger.Logf("Warmup worker %d: error: %v", workerId, err)
				}
			}
		}(i)
	}

	// Feed queries to workers
	for _, query := range queries {
		workChan <- query
	}
	close(workChan)

	wg.Wait()
	logger.Log(fmt.Sprintf("Warmup completed: %d queries executed", len(queries)))
}
