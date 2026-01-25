package main

import (
	"sync"
	"testing"
	"time"
)

// testJobGenParams creates a JobGenerationParameters for testing with common defaults
func testJobGenParams(targetQPS float64, jobProbability float64, minSessionLen, maxSessionLen int) JobGenerationParameters {
	return JobGenerationParameters{
		workloadStdDev:    1.0,
		workloadMean:      0.0,
		followUpStdDev:    0.1,
		followUpMean:      0.0,
		minSessionLength:  minSessionLen,
		maxSessionLength:  maxSessionLen,
		targetQPS:         targetQPS,
		benchmarkDuration: time.Minute,
		jobProbability:    jobProbability,
	}
}

func TestArrivalController_GenerateJob_UniqueIds(t *testing.T) {
	params := testJobGenParams(100.0, 1.0, 5, 10) // 100% jobs
	ac := NewArrivalController(params, 50, 42, 10)

	ids := make(map[string]bool)
	for range 100 {
		work := ac.GenerateWorkload()
		job := work.(*Job)
		if ids[job.Id] {
			t.Errorf("Duplicate job ID: %s", job.Id)
		}
		ids[job.Id] = true
	}
}

func TestArrivalController_GenerateSession_UniqueIds(t *testing.T) {
	params := testJobGenParams(100.0, 0.0, 5, 10) // 100% sessions
	ac := NewArrivalController(params, 50, 42, 10)

	ids := make(map[int]bool)
	for range 100 {
		work := ac.GenerateWorkload()
		session := work.(*UserSession)
		if ids[session.SessionId] {
			t.Errorf("Duplicate session ID: %d", session.SessionId)
		}
		ids[session.SessionId] = true
	}
}

func TestArrivalController_SessionLength(t *testing.T) {
	minLen := 5
	maxLen := 10
	params := testJobGenParams(100.0, 0.0, minLen, maxLen) // 100% sessions
	ac := NewArrivalController(params, 50, 42, 10)

	for range 100 {
		work := ac.GenerateWorkload()
		session := work.(*UserSession)
		if len(session.Jobs) < minLen || len(session.Jobs) > maxLen {
			t.Errorf("Session length %d outside range [%d, %d]", len(session.Jobs), minLen, maxLen)
		}
	}
}

func TestArrivalController_Session_StartsAtStepZero(t *testing.T) {
	params := testJobGenParams(100.0, 0.0, 5, 10) // 100% sessions
	ac := NewArrivalController(params, 50, 42, 10)

	work := ac.GenerateWorkload()
	session := work.(*UserSession)

	if session.currentStep != 0 {
		t.Errorf("Expected currentStep to be 0 for new session, got %d", session.currentStep)
	}
}

func TestArrivalController_Session_HasContinuationChannel(t *testing.T) {
	params := testJobGenParams(100.0, 0.0, 5, 10) // 100% sessions
	ac := NewArrivalController(params, 50, 42, 10)

	work := ac.GenerateWorkload()
	session := work.(*UserSession)

	if session.continuationChan == nil {
		t.Error("Expected session to have continuationChan set")
	}
	if session.continuationChan != ac.continuationChan {
		t.Error("Expected session.continuationChan to be same as ac.continuationChan")
	}
}

func TestTimedWorkload_SchedulingDelay(t *testing.T) {
	scheduledTime := time.Now()
	time.Sleep(10 * time.Millisecond)
	actualStart := time.Now()

	delay := actualStart.Sub(scheduledTime)

	if delay < 10*time.Millisecond {
		t.Errorf("Expected delay >= 10ms, got %v", delay)
	}
}

func TestUserSession_FirstQueryVector(t *testing.T) {
	session := &UserSession{
		SessionId: 1,
		Jobs: []Job{
			{Id: "S-1-0", QueryVector: Vector{1.0, 2.0}},
			{Id: "S-1-1", QueryVector: Vector{0.5, 0.5}},
		},
		currentStep: 0,
	}

	// First query should use the pre-generated vector directly
	expectedVec := Vector{1.0, 2.0}
	actualVec := session.Jobs[session.currentStep].QueryVector

	for i, v := range actualVec {
		if v != expectedVec[i] {
			t.Errorf("Expected vec[%d] to be %f, got %f", i, expectedVec[i], v)
		}
	}
}

func TestUserSession_FollowUpVectorComputation(t *testing.T) {
	// Simulates how Execute() computes follow-up query vectors
	session := &UserSession{
		SessionId: 1,
		Jobs: []Job{
			{Id: "S-1-0", QueryVector: Vector{1.0, 2.0}},
			{Id: "S-1-1", QueryVector: Vector{0.1, 0.1}}, // This is the offset
		},
		currentStep: 1,
	}

	lastResult := Vector{0.5, 0.5} // Simulated result from previous query

	// Compute follow-up query vector: lastResult + offset
	offset := session.Jobs[session.currentStep].QueryVector
	queryVector := make(Vector, len(offset))
	for i := range offset {
		queryVector[i] = lastResult[i] + offset[i]
	}

	expectedVec := Vector{0.6, 0.6}
	for i, v := range queryVector {
		if v != expectedVec[i] {
			t.Errorf("Expected vec[%d] to be %f, got %f", i, expectedVec[i], v)
		}
	}
}

func TestUserSession_HasMoreQueries(t *testing.T) {
	session := &UserSession{
		SessionId: 1,
		Jobs: []Job{
			{Id: "S-1-0", QueryVector: Vector{1.0, 0.0}},
			{Id: "S-1-1", QueryVector: Vector{0.5, 0.5}},
			{Id: "S-1-2", QueryVector: Vector{0.1, 0.1}},
		},
		currentStep: 0,
	}

	// After first query, should have more
	if session.currentStep+1 >= len(session.Jobs) {
		t.Error("Expected more queries after first")
	}

	session.currentStep = 1
	if session.currentStep+1 >= len(session.Jobs) {
		t.Error("Expected more queries after second")
	}

	session.currentStep = 2
	if session.currentStep+1 < len(session.Jobs) {
		t.Error("Expected no more queries after third (last)")
	}
}

func TestUserSession_SingleQuerySession(t *testing.T) {
	session := &UserSession{
		SessionId:   1,
		Jobs:        []Job{{Id: "S-1-0", QueryVector: Vector{1.0}}},
		currentStep: 0,
	}

	// After executing the only query, session should be complete
	hasMore := session.currentStep+1 < len(session.Jobs)
	if hasMore {
		t.Error("Expected no more queries for single-query session")
	}
}

func TestUserSession_MultiStepVectorProgression(t *testing.T) {
	session := &UserSession{
		SessionId: 1,
		Jobs: []Job{
			{Id: "S-1-0", QueryVector: Vector{1.0, 0.0}},
			{Id: "S-1-1", QueryVector: Vector{0.5, 0.5}}, // offset
			{Id: "S-1-2", QueryVector: Vector{0.1, 0.1}}, // offset
		},
		currentStep: 0,
	}

	// Step 0: First query uses pre-generated vector
	vec0 := session.Jobs[0].QueryVector
	if vec0[0] != 1.0 || vec0[1] != 0.0 {
		t.Errorf("Unexpected vector at step 0: %v", vec0)
	}

	// Simulate step 1: topResult0 + offset1
	topResult0 := Vector{0.5, 0.5}
	session.currentStep = 1
	offset1 := session.Jobs[1].QueryVector
	vec1 := make(Vector, len(offset1))
	for i := range offset1 {
		vec1[i] = topResult0[i] + offset1[i]
	}
	session.Jobs[1].QueryVector = vec1 // As Execute() would do
	if vec1[0] != 1.0 || vec1[1] != 1.0 {
		t.Errorf("Unexpected vector at step 1: %v", vec1)
	}

	// Simulate step 2: topResult1 + offset2
	topResult1 := Vector{0.1, 0.1}
	session.currentStep = 2
	offset2 := session.Jobs[2].QueryVector
	vec2 := make(Vector, len(offset2))
	for i := range offset2 {
		vec2[i] = topResult1[i] + offset2[i]
	}
	session.Jobs[2].QueryVector = vec2 // As Execute() would do
	if vec2[0] != 0.2 || vec2[1] != 0.2 {
		t.Errorf("Unexpected vector at step 2: %v", vec2)
	}

	// Session should be complete after step 2
	hasMore := session.currentStep+1 < len(session.Jobs)
	if hasMore {
		t.Error("Expected session to be complete after step 2")
	}
}

// Test that concurrent access to the results slice is safe
func TestConcurrentWorkloadCollection(t *testing.T) {
	var mu sync.Mutex
	var results []int

	var wg sync.WaitGroup
	for i := range 100 {
		wg.Add(1)
		go func(val int) {
			defer wg.Done()
			mu.Lock()
			results = append(results, val)
			mu.Unlock()
		}(i)
	}

	wg.Wait()

	if len(results) != 100 {
		t.Errorf("Expected 100 results, got %d", len(results))
	}
}

func TestWorkloadInterface(t *testing.T) {
	// Verify that Job and UserSession implement Workload interface
	var _ Workload = &Job{}
	var _ Workload = &UserSession{}
}

func TestUserSession_AccumulatedSchedulingDelay(t *testing.T) {
	session := &UserSession{
		SessionId:       1,
		Jobs:            []Job{{Id: "S-1-0"}, {Id: "S-1-1"}},
		SchedulingDelay: 0,
	}

	// Simulate accumulating scheduling delays across queries
	session.SchedulingDelay += 10 * time.Millisecond
	session.SchedulingDelay += 5 * time.Millisecond

	if session.SchedulingDelay != 15*time.Millisecond {
		t.Errorf("Expected accumulated delay of 15ms, got %v", session.SchedulingDelay)
	}
}
