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
	ac := NewArrivalController(params, 50, 42)

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
	ac := NewArrivalController(params, 50, 42)

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
	ac := NewArrivalController(params, 50, 42)

	for range 100 {
		work := ac.GenerateWorkload()
		session := work.(*UserSession)
		if len(session.jobs) < minLen || len(session.jobs) > maxLen {
			t.Errorf("Session length %d outside range [%d, %d]", len(session.jobs), minLen, maxLen)
		}
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

func TestUserSessionNextQuery_FirstCallReturnsInitialVector(t *testing.T) {
	session := UserSession{
		SessionId:   1,
		jobs:        []Job{{Id: "S-1-0", QueryVector: Vector{1.0, 2.0}}, {Id: "S-1-1", QueryVector: Vector{0.5, 0.5}}},
		currentStep: 0,
	}

	vec, hasNext := session.NextQuery(nil)

	if !hasNext {
		t.Error("Expected hasNext to be true")
	}

	expectedVec := Vector{1.0, 2.0}
	for i, v := range vec {
		if v != expectedVec[i] {
			t.Errorf("Expected vec[%d] to be %f, got %f", i, expectedVec[i], v)
		}
	}

	if session.currentStep != 1 {
		t.Errorf("Expected currentStep to be 1, got %d", session.currentStep)
	}
}

func TestUserSessionNextQuery_FollowUpVector(t *testing.T) {
	session := UserSession{
		SessionId:   1,
		jobs:        []Job{{Id: "S-1-0", QueryVector: Vector{1.0, 2.0}}, {Id: "S-1-1", QueryVector: Vector{0.1, 0.1}}},
		currentStep: 1,
	}

	vec, hasNext := session.NextQuery(Vector{0.5, 0.5})

	if !hasNext {
		t.Error("Expected hasNext to be true")
	}

	// The offset should be lastResult + jobs[1].QueryVector
	expectedVec := Vector{0.6, 0.6}
	for i, v := range vec {
		if v != expectedVec[i] {
			t.Errorf("Expected vec[%d] to be %f, got %f", i, expectedVec[i], v)
		}
	}

	if session.currentStep != 2 {
		t.Errorf("Expected currentStep to be 2, got %d", session.currentStep)
	}
}

func TestUserSessionNextQuery_CompletedSession(t *testing.T) {
	session := UserSession{
		SessionId:   1,
		jobs:        []Job{{Id: "S-1-0", QueryVector: Vector{1.0}}},
		currentStep: 1,
	}

	vec, hasNext := session.NextQuery(Vector{0.5})

	if hasNext {
		t.Error("Expected hasNext to be false when session exhausted")
	}
	if vec != nil {
		t.Error("Expected vec to be nil when session exhausted")
	}
}

func TestUserSessionNextQuery_FullSession(t *testing.T) {
	session := UserSession{
		SessionId: 1,
		jobs: []Job{
			{Id: "S-1-0", QueryVector: Vector{1.0, 0.0}},
			{Id: "S-1-1", QueryVector: Vector{0.5, 0.5}},
			{Id: "S-1-2", QueryVector: Vector{0.1, 0.1}},
		},
		currentStep: 0,
	}

	// initial vector
	vec0, ok0 := session.NextQuery(nil)
	if !ok0 {
		t.Error("Expected ok to be true for step 0")
	}
	if vec0[0] != 1.0 || vec0[1] != 0.0 {
		t.Errorf("Unexpected vector at step 0: %v", vec0)
	}

	// lastResult + offset
	vec1, ok1 := session.NextQuery(Vector{0.5, 0.5})
	if !ok1 {
		t.Error("Expected ok to be true for step 1")
	}
	if vec1[0] != 1.0 || vec1[1] != 1.0 {
		t.Errorf("Unexpected vector at step 1: %v", vec1)
	}

	// lastResult + offset
	vec2, ok2 := session.NextQuery(Vector{0.1, 0.1})
	if !ok2 {
		t.Error("Expected ok to be true for step 2")
	}
	if vec2[0] != 0.2 || vec2[1] != 0.2 {
		t.Errorf("Unexpected vector at step 2: %v", vec2)
	}

	// Session end
	_, ok3 := session.NextQuery(Vector{0.0, 0.0})
	if ok3 {
		t.Error("Expected ok to be false when session exhausted")
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
