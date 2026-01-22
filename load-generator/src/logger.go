package main

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
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
	jobFormat     = "timestamp,jobId,isUserSession,sessionId,step,queryVector,topResultIds,latencyMus,schedulingDelayMus\n"
	sessionFormat = "timestamp,sessionId,numSteps,totalDurationMus,schedulingDelayMus\n"
)

// outputDir holds the current output directory, set by SetOutputDir
var outputDir = "output"

// SetOutputDir sets the output directory for all log files
func SetOutputDir(dir string) {
	outputDir = dir
}

// GetOutputDir returns the current output directory
func GetOutputDir() string {
	return outputDir
}

func ensureOutputDir() error {
	err := os.Mkdir(outputDir, 0755)
	if err != nil && !os.IsExist(err) {
		return err
	}
	return nil
}

// outputPath prefixes the output directory to create a full file path.
func outputPath(filename string) string {
	return filepath.Join(outputDir, filename)
}

func NewLogger(prefix string) (*Logger, error) {
	if err := ensureOutputDir(); err != nil {
		return nil, err
	}

	logFile, err := os.OpenFile(
		outputPath(fmt.Sprintf("%s-%s.txt", prefix, basePath)),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0644,
	)
	if err != nil {
		return nil, err
	}
	jobFile, err := os.OpenFile(
		outputPath(fmt.Sprintf("%s-jobs.csv", prefix)),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0644,
	)
	if err != nil {
		return nil, err
	}
	sessionFile, err := os.OpenFile(
		outputPath(fmt.Sprintf("%s-%s-session.csv", prefix, basePath)),
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
	logEntry := fmt.Sprintf(format, args...)
	l.Log(logEntry)
	fmt.Println(logEntry)
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
		job.Latency.Microseconds(),
		job.SchedulingDelay.Microseconds(),
	)
	l.jobLogFile.WriteString(logEntry)
}

func (l *Logger) LogSession(session *UserSession) {
	logEntry := fmt.Sprintf(
		"%s,%d,%d,%d,%d\n",
		session.StartTimestamp.Format(time.DateTime),
		session.SessionId,
		len(session.jobs),
		session.Latency.Microseconds(),
		session.SchedulingDelay.Microseconds(),
	)
	l.sessionLogFile.WriteString(logEntry)
}

func (l *Logger) LogDataRows(data []DataRow) error {
	gobFile, err := os.Create(outputPath("data-rows.gob"))
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(gobFile)
	err = encoder.Encode(data)
	return err
}

func (l *Logger) LogEnhancedResults(results []EnhancedJobResult) error {
	return parquet.WriteFile(outputPath("enhanced-results.parquet"), results)
}

func (l *Logger) Close() {
	l.logFile.Close()
	l.jobLogFile.Close()
	l.sessionLogFile.Close()
}
