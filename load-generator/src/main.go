package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

type ConstructionIndexParameters struct {
	distanceMetric string
	M              int
	efConstruction int
}

type JobGenerationParameters struct {
	workloadStdDev    float32
	workloadMean      float32
	followUpStdDev    float32
	followUpMean      float32
	minSessionLength  int
	maxSessionLength  int
	targetQPS         float64 // Target queires per second
	benchmarkDuration time.Duration
	jobProbability    float64 // Probability of generating a Job vs UserSession (0.0-1.0)
}

type Config struct {
	milvusAddr          string
	dbName              string
	collection          string
	idFieldName         string
	vecFieldName        string
	fieldName           string
	dim                 int
	concurrency         int
	ef                  int
	k                   int
	insertBatchSize     int
	numberWarmupQueries int
	dataFile            string
	indexParameters     ConstructionIndexParameters
	jobGenParams        JobGenerationParameters
}

const milvusPort = "19530"

// getMilvusAddr returns the Milvus address from environment variable MILVUS_IP or localhost as fallback.
func getMilvusAddr() string {
	ip := os.Getenv("MILVUS_IP")
	if ip == "" {
		fmt.Println("MILVUS_IP not set, defaulting to localhost")
		ip = "localhost"
	}
	return ip + ":" + milvusPort
}

var config Config = Config{
	milvusAddr:          getMilvusAddr(),
	dbName:              "benchmark",
	collection:          "benchmarkData",
	idFieldName:         "id",
	vecFieldName:        "vector",
	fieldName:           "word",
	concurrency:         50,
	ef:                  400, // how many neighbors to evaluate during the search
	k:                   10,  // number of results returned from the query
	insertBatchSize:     1000,
	numberWarmupQueries: 5000,
	jobGenParams: JobGenerationParameters{
		workloadStdDev:    7.5,
		workloadMean:      0.0,
		followUpStdDev:    0.15,
		followUpMean:      1.25,
		minSessionLength:  5,
		maxSessionLength:  50,
		targetQPS:         100.0,
		benchmarkDuration: 30 * time.Minute,
		jobProbability:    0.85,
	},
	indexParameters: ConstructionIndexParameters{
		distanceMetric: "L2", // euclidean distance (constant)
	},
}

var validDatasetIds = map[int]bool{50: true, 100: true, 200: true}

func parseArgs() (configId int, dimId int, recallAfterBenchmark bool, err error) {
	if len(os.Args) < 3 || len(os.Args) > 4 {
		return 0, 0, true, fmt.Errorf(`usage: %s <config_id> <dataset_id> <offline_recall>
			config_id:  index configuration number (1-3)
			dataset_id: dataset dimensionality (50, 100, 200)
			Optional: recall_after_benchmark (true/false) whether to calculate recall directly after benchmark execution (defaults to true)`,
			os.Args[0])
	}

	configId, err = strconv.Atoi(os.Args[1])
	if err != nil || configId < 1 || configId > 3 {
		return 0, 0, true, fmt.Errorf("invalid config_id: must be a number between 1 and 3")
	}
	dimId, err = strconv.Atoi(os.Args[2])
	if err != nil || !validDatasetIds[dimId] {
		return 0, 0, true, fmt.Errorf("invalid dimensionality: must be one of [50, 100, 200]")
	}

	recallAfterBenchmark, err = strconv.ParseBool(os.Args[3])
	if err != nil {
		recallAfterBenchmark = true // default to true if not provided or invalid
	}

	return
}

func main() {
	/* Parse CLI arguments and load configurations */
	configId, dimId, recallAfterBenchmark, err := parseArgs()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	err = LoadIndexConfig(configId, &config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load index configuration: %v\n", err)
		os.Exit(1)
	}
	err = LoadDimConfig(dimId, &config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load dataset configuration: %v\n", err)
		os.Exit(1)
	}
	SetOutputDir(fmt.Sprintf("output-config%d-dim%d", configId, dimId))

	/* Initialize Benchmark */
	logger, err := NewLogger("main")
	if err != nil {
		panic(err)
	}
	defer logger.Close()
	logger.Logf("Benchmark started with config Id %d, dataset dimensionality %d:\n%+v", configId, dimId, config)

	ctx := context.Background()
	logger.Logf("Connecting to Milvus at %s...", config.milvusAddr)
	c, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:  config.milvusAddr,
		Username: "root",
		Password: "Milvus",
	})
	if err != nil {
		panic(err)
	}
	defer c.Close(ctx) // close connection after experiments are run
	logger.Log("Successfully connected")

	datasource := DataReader{config.dataFile}

	/* Prepare the benchmark: create collection, insert data, create index */
	err = Prepare(
		c,
		config.dbName,
		config.collection,
		config.idFieldName,
		config.vecFieldName,
		config.dim,
		config.fieldName,
		config.indexParameters,
		config.insertBatchSize,
		datasource,
	)
	if err != nil {
		panic(err)
	}

	/* Warmup */
	err = Warmup(
		c,
		config.numberWarmupQueries,
		config.dim,
		config.collection,
		config.vecFieldName,
		config.k,
	)
	if err != nil {
		panic(err)
	}

	/* Execute Benchmark */
	jobs, sessions, err := ExecuteBenchmark(
		c,
		config.collection,
		config.vecFieldName,
		datasource,
		config.dim,
		config.jobGenParams,
		config.k,
		config.concurrency,
	)
	if err != nil {
		panic(err)
	}

	logger.Log("Benchmark completed successfully")

	/* Cleanup */
	logger.Log("Cleaning up: deleting collection and database...")
	err = Cleanup(c, config.dbName, config.collection)
	if err != nil {
		logger.Log(err.Error())
	}

	/* Enhance Results by calculating recall */
	if (recallAfterBenchmark) {
	logger.Log("Calculating recall...")
		err = Collection(datasource, jobs, sessions)
		if err != nil {
			panic(err)
		}
	} else {
		logger.Log("Saving jobs and sessions in gob format for offline recall calculation...")
		err = logger.LogJobsAndSessionsGob(jobs, sessions)
		if err != nil {
			panic(err)
		}
	}

	logger.Log("Benchmark finished.")
}
