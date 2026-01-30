package main

import (
	"fmt"
	"os"
	"time"
	"encoding/gob"

	"github.com/parquet-go/parquet-go"
)

type Vector []float32

type DataRow struct {
	Id     int64
	Vector Vector
	Word   string
}

type Job struct {
	Id              string // Unique identifier (for independent jobs: "J-{index}", for session jobs: "S-{sessionId}-{step}")
	QueryVector     Vector
	ResultIds       []int64
	Latency         time.Duration
	StartTimestamp  time.Time
	SchedulingDelay time.Duration // Time between scheduled arrival and actual execution start
}

type UserSession struct {
	SessionId       int
	Jobs            []Job
	StartTimestamp  time.Time
	Duration        time.Duration
	SchedulingDelay time.Duration // Time between scheduled arrival and actual execution start

	currentStep      int
	continuationChan chan *UserSession
}

func main() {
	basePath := os.Args[1]
	if (basePath == "") {
		panic(fmt.Errorf("basePath is required"))
	}

	entries, err := os.ReadDir(basePath)
	if err != nil {
		panic(err)
	}

	for _, entry := range entries {
		recall(basePath, entry)
	}

}

func recall(basePath string, entry os.DirEntry) {
	if (!entry.IsDir()) {
		return
	}
	dataRows, err:= readDataRows(basePath, entry)
	if err != nil {
		return
	}
	jobs, sessions, err := readJobsAndSessions(basePath, entry)
	if err != nil {
		return
	}

	sessionJobs := mapSessionsToJobs(sessions)
	allJobs := append(jobs, sessionJobs...)

	enhancedResults := EnhanceJobResults(dataRows, allJobs)
	err = parquet.WriteFile(fmt.Sprintf("%s/%s/enhanced-results.parquet", basePath, entry.Name()), enhancedResults)
	if err != nil {
		fmt.Printf("failed to write enhanced-results.parquet for %s: %v\n", entry.Name(), err)
	}
}

func mapSessionsToJobs(sessions []UserSession) (jobs []Job) {
	for _, session := range sessions {
		jobs = append(jobs, session.Jobs...)
	}
	return
}

func readDataRows(basePath string, entry os.DirEntry) ([]DataRow, error) {
	dataRows, err := os.Open(fmt.Sprintf("%s/%s/data-rows.gob", basePath, entry.Name()))
	if err != nil {
		fmt.Printf("failed to open data-rows.gob for %s: %v\n", entry.Name(), err)
		return nil, err
	}
	defer dataRows.Close()
	decoder := gob.NewDecoder(dataRows)
	var rows []DataRow
	err = decoder.Decode(&rows)
	return rows, err
}

func readJobsAndSessions(basePath string, entry os.DirEntry) ([]Job, []UserSession, error) {
	gobFile, err := os.Open(fmt.Sprintf("%s/%s/jobs-sessions.gob", basePath, entry.Name()))
	if err != nil {
		return nil, nil, err
	}

	var decoded struct {
		Jobs     []Job
		Sessions []UserSession
	}
	decoder := gob.NewDecoder(gobFile)
	err = decoder.Decode(&decoded)
	return decoded.Jobs, decoded.Sessions, err
}
