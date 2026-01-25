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
	jobGenParams     JobGenerationParameters
	dim              int // dim is not part of the JobGenerationParameters because it is used in many places
	gen              *rand.Rand
	continuationChan chan *UserSession

	// Counters for Id generation
	jobCounter     int
	sessionCounter int
}

type TimedWorkload struct {
	Work          Workload
	ScheduledTime time.Time // Captures the wait time until a worker was able to pick up the work
}

// Workload is the interface for executable benchmark work units.
type Workload interface {
	Execute(
		ctx context.Context,
		c *milvusclient.Client,
		collection string,
		vecFieldName string,
		dim int,
		k int,
		logger *Logger,
		schedulingDelay time.Duration,
	) (Workload, error)
}

// Job is a single kNN search query
type Job struct {
	Id              string // Unique identifier (for independent jobs: "J-{index}", for session jobs: "S-{sessionId}-{step}")
	QueryVector     Vector
	ResultIds       []int64
	Latency         time.Duration
	StartTimestamp  time.Time
	SchedulingDelay time.Duration // Time between scheduled arrival and actual execution start
}

/**
* UserSession simulates a somewhat realistic user behavior with sequential, dependent queries.
* Each session starts with a random query vector, then subsequent queries are based on the top
* result from the previous query plus a small random offset to simulate attention-based drift.
*
* Job Ids within a session are encoded as "S-{sessionId}-{stepIndex}".
 */
type UserSession struct {
	SessionId       int
	Jobs            []Job
	StartTimestamp  time.Time
	Duration        time.Duration
	SchedulingDelay time.Duration // Time between scheduled arrival and actual execution start

	currentStep      int
	continuationChan chan *UserSession
}

func NewArrivalController(
	jobGenParams JobGenerationParameters,
	dim int,
	seed int64,
	continuationBufferSize int,
) *ArrivalController {
	continuationChan := make(chan *UserSession, continuationBufferSize)

	return &ArrivalController{
		jobGenParams:     jobGenParams,
		dim:              dim,
		gen:              rand.New(rand.NewSource(seed)),
		continuationChan: continuationChan,
		jobCounter:       0,
		sessionCounter:   0,
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

// GenerateWorkload creates either a Job or SessionQuery (first query of a session) based on jobProbability
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
		// The first query uses the same distribution as independent jobs, follow-up offsets use a different distribution
		if j == 0 {
			query = GenerateVector(ac.gen, ac.dim, ac.jobGenParams.workloadStdDev, ac.jobGenParams.workloadMean)
		} else {
			query = GenerateVector(ac.gen, ac.dim, ac.jobGenParams.followUpStdDev, ac.jobGenParams.followUpMean)
		}
		jobId := fmt.Sprintf("S-%d-%d", ac.sessionCounter, j)
		jobs[j] = Job{Id: jobId, QueryVector: query}
	}

	session := &UserSession{
		SessionId:        ac.sessionCounter,
		Jobs:             jobs,
		currentStep:      0,
		continuationChan: ac.continuationChan,
	}
	ac.sessionCounter++
	return session
}

/**
* ExecuteWorkloadPoisson runs workloads concurrently with Poisson-distributed arrivals.
* It returns the executed Jobs and UserSessions to enable recall analysis.
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
				schedulingDelay := actualStart.Sub(timedWork.ScheduledTime)

				res, err := timedWork.Work.Execute(
					ctx,
					c,
					collection,
					vecFieldName,
					dim,
					k,
					logger,
					schedulingDelay,
				)
				if err != nil && err != context.Canceled { // Errors are expected on benchmark end
					logger.Logf("Worker %d: error executing work: %v", workerId, err)
					continue
				}

				if res == nil {
					// Continuation enqueued, skip collecting result
					continue
				}

				// Collect results
				mu.Lock()
				switch r := res.(type) {
				case *Job:
					executedJobs = append(executedJobs, *r)
				case *UserSession:
					executedSessions = append(executedSessions, *r)
				}
				mu.Unlock()
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

			// Prioritize continuations over new workloads
			var work Workload
			select {
			case continuation := <-ac.continuationChan:
				work = continuation
			default:
				work = ac.GenerateWorkload()
			}

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

	// Note: ac.continuationChan may still have pending sessions that won't complete
	logger.Logf("Executed %d jobs and %d sessions", len(executedJobs), len(executedSessions))
	return executedJobs, executedSessions
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

// Execute runs a single session query and enqueues the next query if the session continues.
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
	select {
	case <-ctx.Done():
		us.Duration = time.Since(us.StartTimestamp)
		logger.Logf("Session %d cancelled after %d of %d steps", us.SessionId, us.currentStep, len(us.Jobs))
		return nil, ctx.Err()
	default:
	}

	job := &us.Jobs[us.currentStep]
	us.SchedulingDelay += schedulingDelay // Accumulate scheduling delays
	if us.currentStep == 0 {
		// For the first query, record session start time and scheduling delay
		us.StartTimestamp = time.Now()
	}

	// Execute the k-NN search
	jobStart := time.Now()
	searchRes, err := c.Search(ctx,
		milvusclient.NewSearchOption(
			collection,
			k,
			[]entity.Vector{entity.FloatVector(job.QueryVector)},
		).WithANNSField(vecFieldName).
			WithOutputFields(vecFieldName), // Need vector field for computing next query
	)

	job.Latency = time.Since(jobStart)
	job.StartTimestamp = jobStart
	job.SchedulingDelay = schedulingDelay

	if err != nil {
		// On error, return partial session
		us.Duration = time.Since(us.StartTimestamp)
		return us, err
	}

	if len(searchRes) != 1 {
		logger.Logf("Unexpected number of result sets: %d", len(searchRes))
	}

	var topResult Vector
	for _, resultSet := range searchRes {
		job.ResultIds = resultSet.IDs.FieldData().GetScalars().GetLongData().Data
		vectors := resultSet.GetColumn(vecFieldName)
		if vectors == nil {
			logger.Logf("Session %d: No vector field '%s' in search result", us.SessionId, vecFieldName)
			continue
		}
		// Don't ask why but this concatenates all the vectors so we must slice to get the first one
		combinedVector := vectors.FieldData().GetVectors().GetFloatVector().Data
		topResult = combinedVector[:dim]
	}

	logger.LogJob(job, us.SessionId, us.currentStep)

	// Check if more queries remain in the session
	if us.currentStep+1 < len(us.Jobs) {
		if topResult == nil {
			// Cannot compute next query without top result vector, end session early
			logger.Logf("Session %d: No vector field '%s' in result, ending session early at step %d",
				us.SessionId, vecFieldName, us.currentStep)
			us.Duration = time.Since(us.StartTimestamp)
			logger.LogSession(us)
			return us, nil
		}

		us.currentStep++
		// Compute next query vector based on last result + offset
		offset := us.Jobs[us.currentStep].QueryVector
		nextQuery := make(Vector, dim)
		for i := range dim {
			nextQuery[i] = topResult[i] + offset[i]
		}
		us.Jobs[us.currentStep].QueryVector = nextQuery

		// Enqueue continuation
		select {
		case us.continuationChan <- us:
			return nil, nil
		case <-ctx.Done():
			// Context cancelled, return partial session
			us.Duration = time.Since(us.StartTimestamp)
			return us, ctx.Err()
		}
	}

	// Session complete
	us.Duration = time.Since(us.StartTimestamp)
	logger.LogSession(us)
	return us, nil
}
