package main

func MapSessionsToJobs(sessions []UserSession) (jobs []Job) {
	for _, session := range sessions {
		jobs = append(jobs, session.Jobs...)
	}
	return
}

func Collection(datasource DataSource, jobs []Job, sessions []UserSession) error {
	logger, err := NewLogger("collection")
	if err != nil {
		return err
	}
	defer logger.Close()

	rows, err := datasource.ReadDataRows()
	if err != nil {
		return err
	}

	sessionJobs := MapSessionsToJobs(sessions)
	allJobs := append(jobs, sessionJobs...)

	enhancedResults := EnhanceJobResults(rows, allJobs)
	return logger.LogEnhancedResults(enhancedResults)
}
