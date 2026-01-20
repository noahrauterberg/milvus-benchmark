package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

/**
* ArrivalController manages Poisson process arrivals for workload generation.
* It generates workloads on-demand with exponentially distributed inter-arrival times.
 */
type ArrivalController struct {
	jobGenParams JobGenerationParameters
	dim          int
	gen          *rand.Rand

	// Counters for ID generation (accessed only by arrival goroutine)
	jobCounter     int
	sessionCounter int
}

func NewArrivalController(
	jobGenParams JobGenerationParameters,
	dim int,
	seed int64,
) *ArrivalController {
	return &ArrivalController{
		jobGenParams:   jobGenParams,
		dim:            dim,
		gen:            rand.New(rand.NewSource(seed)),
		jobCounter:     0,
		sessionCounter: 0,
	}
}

func (ac *ArrivalController) NextSleepDuration() time.Duration {
	// Exponential distribution: -ln(U) / lambda where U ~ Uniform(0,1)
	u := ac.gen.Float64()
	// Avoid log(0)
	for u == 0 {
		u = ac.gen.Float64()
	}
	interval := -math.Log(u) / ac.jobGenParams.targetQPS
	return time.Duration(interval * float64(time.Second))
}

// GenerateWorkload creates either a Job or UserSession based on jobProbability
func (ac *ArrivalController) GenerateWorkload() Workload {
	if ac.gen.Float64() < ac.jobGenParams.jobProbability {
		return ac.generateJob()
	}
	return ac.generateSession()
}

func (ac *ArrivalController) generateJob() *Job {
	query := GenerateVector(ac.gen, ac.dim, ac.jobGenParams.workloadStdDev, ac.jobGenParams.workloadMean)
	jobId := fmt.Sprintf("J-%d", ac.jobCounter)
	ac.jobCounter++
	return &Job{Id: jobId, QueryVector: query}
}

func (ac *ArrivalController) generateSession() *UserSession {
	minLen := ac.jobGenParams.minSessionLength
	maxLen := ac.jobGenParams.maxSessionLength
	sessionLength := ac.gen.Intn(maxLen-minLen+1) + minLen
	jobs := make([]Job, sessionLength)
	for j := range sessionLength {
		var query []float32
		if j == 0 {
			query = GenerateVector(ac.gen, ac.dim, ac.jobGenParams.workloadStdDev, ac.jobGenParams.workloadMean)
		} else {
			query = GenerateVector(ac.gen, ac.dim, ac.jobGenParams.followUpStdDev, ac.jobGenParams.followUpMean)
		}
		jobId := fmt.Sprintf("S-%d-%d", ac.sessionCounter, j)
		jobs[j] = Job{Id: jobId, QueryVector: query}
	}
	session := &UserSession{
		SessionId:   ac.sessionCounter,
		jobs:        jobs,
		currentStep: 0,
	}
	ac.sessionCounter++
	return session
}

type TimedWorkload struct {
	Work          Workload
	ScheduledTime time.Time // Captures the wait time until a worker was able to pick up the work
}

/**
* ExecuteWorkload runs workload concurrently with Poisson-distributed arrivals.
* It returns the executed Jobs and UserSessions to enable recall analysis
 */
func ExecuteWorkloadPoisson(
	ac *ArrivalController,
	c *milvusclient.Client,
	collection string,
	vecFieldName string,
	dim int,
	k int,
	logger *Logger,
	numWorkers int,
) ([]Job, []UserSession) {
	workChan := make(chan TimedWorkload, numWorkers*2)

	// Allows to communicate benchmark end to workers
	ctx, cancel := context.WithCancel(context.Background())

	var mu sync.Mutex
	var executedJobs []Job
	var executedSessions []UserSession

	/* Worker goroutines */
	var wg sync.WaitGroup
	for i := range numWorkers {
		wg.Add(1)
		go func(workerId int) {
			defer wg.Done()
			for timedWork := range workChan {
				actualStart := time.Now()
				// Scheduling delay
				schedulingDelay := actualStart.Sub(timedWork.ScheduledTime)

				result, err := timedWork.Work.Execute(ctx, c, collection, vecFieldName, dim, k, logger, schedulingDelay)
				if err != nil && err != context.Canceled { // Errors are expected on benchmark end
					logger.Logf("Worker %d: error executing work: %v", workerId, err)
				}

				// Collect results
				if result != nil {
					mu.Lock()
					switch r := result.(type) {
					case *Job:
						executedJobs = append(executedJobs, *r)
					case *UserSession:
						executedSessions = append(executedSessions, *r)
					}
					mu.Unlock()
				}
			}
		}(i)
	}

	/* Arrival goroutine */
	go func() {
		defer close(workChan)
		startTime := time.Now()
		duration := ac.jobGenParams.benchmarkDuration

		for {
			sleepTime := ac.NextSleepDuration()
			time.Sleep(sleepTime)

			// Check if the benchmark duration is already over
			if time.Since(startTime) >= duration {
				logger.Log("Benchmark duration reached, stopping arrivals")
				cancel()
				return
			}

			// Generate and send workload
			work := ac.GenerateWorkload()
			scheduledTime := time.Now()

			select {
			case workChan <- TimedWorkload{Work: work, ScheduledTime: scheduledTime}:
			case <-time.After(1 * time.Second):
				logger.Log("Warning: work channel full, dropping workload")
			}
		}
	}()

	// Wait for all workers to complete remaining work
	wg.Wait()

	logger.Logf("Executed %d jobs and %d sessions", len(executedJobs), len(executedSessions))
	return executedJobs, executedSessions
}

// Workload is the interface for executable benchmark work units (i.e. Jobs and UserSessions).
type Workload interface {
	Execute(ctx context.Context, c *milvusclient.Client, collection string, vecFieldName string, dim int, k int, logger *Logger, schedulingDelay time.Duration) (Workload, error)
}

// Job is a single kNN search query
type Job struct {
	Id              string // Unique identifier (for independent jobs, this follows: "J-{index}")
	QueryVector     Vector
	ResultIds       []int64
	Latency         time.Duration
	StartTimestamp  time.Time
	SchedulingDelay time.Duration // Time between scheduled arrival and actual execution start
}

// Execute performs the k-NN search for this job and records metrics.
func (j *Job) Execute(
	ctx context.Context,
	c *milvusclient.Client,
	collection string,
	vecFieldName string,
	dim int,
	k int,
	logger *Logger,
	schedulingDelay time.Duration,
) (Workload, error) {
	// Check if context is cancelled before starting
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	j.SchedulingDelay = schedulingDelay
	start := time.Now()

	searchRes, err := c.Search(ctx,
		milvusclient.NewSearchOption(
			collection,
			k,
			[]entity.Vector{entity.FloatVector(j.QueryVector)},
		).WithANNSField(vecFieldName),
	)
	j.Latency = time.Since(start)
	j.StartTimestamp = start
	if err != nil {
		return nil, err
	}

	if len(searchRes) != 1 {
		logger.Logf("Unexpected number of result sets: %d", len(searchRes))
	}
	for _, resultSet := range searchRes {
		j.ResultIds = resultSet.IDs.FieldData().GetScalars().GetLongData().Data
	}
	logger.LogJob(j, -1, -1) // -1 indicates not part of a session
	return j, nil
}

/**
* UserSession simulates a somewhat realistic user behavior with sequential, dependent queries.
* Each session starts with a random query vector, then subsequent queries are based on the top result
* from the previous query plus a small random offset to simulate an attention based vector drift.
*
* Job Ids within a session are formatted as "S-{sessionId}-{stepIndex}".
 */
type UserSession struct {
	SessionId       int
	jobs            []Job // Queries in this session (initially only the offset vectors)
	currentStep     int
	StartTimestamp  time.Time
	Latency         time.Duration // Total session duration
	SchedulingDelay time.Duration // Time between scheduled arrival and actual execution start
}

// Execute runs all queries in the user session sequentiallym respecting context cancellation.
func (us *UserSession) Execute(
	ctx context.Context,
	c *milvusclient.Client,
	collection string,
	vecFieldName string,
	dim int,
	k int,
	logger *Logger,
	schedulingDelay time.Duration,
) (Workload, error) {
	us.SchedulingDelay = schedulingDelay
	start := time.Now()
	us.StartTimestamp = start

	for queryVector, hasNext := us.NextQuery(nil); hasNext; {
		select {
		case <-ctx.Done():
			// Log partial session completion
			us.Latency = time.Since(start)
			logger.Logf("Session %d terminated early after %d/%d steps", us.SessionId, us.currentStep, len(us.jobs))
			return us, ctx.Err()
		default:
		}

		jobStart := time.Now()
		searchRes, err := c.Search(ctx,
			milvusclient.NewSearchOption(
				collection,
				k,
				[]entity.Vector{entity.FloatVector(queryVector)},
			).WithANNSField(vecFieldName),
		)
		if err != nil {
			us.Latency = time.Since(start)
			return us, err
		}

		// Record job timing (-1 because currentStep was incremented in NextQuery)
		us.jobs[us.currentStep-1].Latency = time.Since(jobStart)
		us.jobs[us.currentStep-1].StartTimestamp = jobStart

		if len(searchRes) != 1 {
			logger.Logf("Unexpected number of result sets: %d", len(searchRes))
		}
		for _, resultSet := range searchRes {
			us.jobs[us.currentStep-1].ResultIds = resultSet.IDs.FieldData().GetScalars().GetLongData().Data
			vectors := resultSet.GetColumn(vecFieldName)
			// Don't ask why but this concatenates all the vectors so we must slice to get the first one
			combinedVector := vectors.FieldData().GetVectors().GetFloatVector().Data
			// TODO: prevent selecting the same vector every time
			topResult := combinedVector[:dim]
			queryVector, hasNext = us.NextQuery(topResult)
		}

		logger.LogJob(&us.jobs[us.currentStep-1], us.SessionId, us.currentStep-1)
	}

	us.Latency = time.Since(start)
	logger.LogSession(us)
	return us, nil
}

/**
* NextQuery computes and returns the next query vector for the session
* based on the last result and the offset.
* Returns (vector, true) if there's a next query, or (nil, false) if session is complete.
 */
func (us *UserSession) NextQuery(lastResult Vector) (Vector, bool) {
	if us.currentStep == 0 {
		us.currentStep++
		return us.jobs[0].QueryVector, true
	}

	// Subsequent queries - compute based on last result + offset
	if us.currentStep < len(us.jobs) {
		offset := us.jobs[us.currentStep].QueryVector
		for i := range lastResult {
			offset[i] += lastResult[i]
		}
		us.jobs[us.currentStep].QueryVector = offset
		us.currentStep++
		return offset, true
	}
	return nil, false
}
