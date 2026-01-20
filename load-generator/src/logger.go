package main

import (
	"encoding/gob"
	"fmt"
	"os"
	"time"

	"github.com/parquet-go/parquet-go"
)

type Logger struct {
	logFile        os.File
	jobLogFile     os.File
	sessionLogFile os.File
}

const (
	basePath = "log"
	// CSV format for logging queries
	jobFormat     = "timestamp,jobId,isUserSession,sessionId,step,queryVector,topResultIds,latencyMs,schedulingDelayMs\n"
	sessionFormat = "timestamp,sessionId,numSteps,totalDurationMs,schedulingDelayMs\n"
)

func NewLogger(prefix string) (*Logger, error) {
	logFile, err := os.OpenFile(
		fmt.Sprintf("%s-%s.txt", prefix, basePath),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0644,
	)
	if err != nil {
		return nil, err
	}
	jobFile, err := os.OpenFile(
		fmt.Sprintf("%s-jobs.csv", prefix),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0644,
	)
	if err != nil {
		return nil, err
	}
	sessionFile, err := os.OpenFile(
		fmt.Sprintf("%s-%s-session.csv", prefix, basePath),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0644,
	)
	if err != nil {
		return nil, err
	}

	jobFile.WriteString(jobFormat)
	sessionFile.WriteString(sessionFormat)

	return &Logger{
		logFile:        *logFile,
		jobLogFile:     *jobFile,
		sessionLogFile: *sessionFile,
	}, nil
}

func (l *Logger) Log(msg string) {
	timestamp := time.Now().Format(time.DateTime)
	logEntry := fmt.Sprintf("[%s] - %s\n", timestamp, msg)
	l.logFile.WriteString(logEntry)
}

func (l *Logger) Logf(format string, args ...any) {
	l.Log(fmt.Sprintf(format, args...))
}

// LogJob logs the details of a Job in CSV format.
func (l *Logger) LogJob(job *Job, sessionId int, step int) {
	var isSession = sessionId >= 0 && step >= 0
	logEntry := fmt.Sprintf(
		"%s,%s,%t,%d,%d,\"%v\",\"%v\",%d,%d\n",
		job.StartTimestamp.Format(time.DateTime),
		job.Id,
		isSession,
		sessionId,
		step,
		job.QueryVector,
		job.ResultIds,
		job.Latency.Milliseconds(),
		job.SchedulingDelay.Milliseconds(),
	)
	l.jobLogFile.WriteString(logEntry)
}

func (l *Logger) LogSession(session *UserSession) {
	logEntry := fmt.Sprintf(
		"%s,%d,%d,%d,%d\n",
		session.StartTimestamp.Format(time.DateTime),
		session.SessionId,
		len(session.jobs),
		session.Latency.Milliseconds(),
		session.SchedulingDelay.Milliseconds(),
	)
	l.sessionLogFile.WriteString(logEntry)
}

func (l *Logger) LogDataRows(data []DataRow) error {
	gobFile, err := os.Create("data-rows.gob")
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(gobFile)
	err = encoder.Encode(data)
	return err
}

func (l *Logger) LogEnhancedResults(results []EnhancedJobResult) error {
	return parquet.WriteFile("enhanced-results.parquet", results)
}

func (l *Logger) Close() {
	l.logFile.Close()
	l.jobLogFile.Close()
	l.sessionLogFile.Close()
}
