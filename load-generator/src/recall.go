package main

import (
	"runtime"
	"sync"
)

// EnhancedJobResult extends Job with the calculated recall metric.
type EnhancedJobResult struct {
	Job
	Recall float64
}

/**
* Strictly speaking, this is not the Euclidean distance but squared Euclidean distance
* However, since we only care about relative distances, we may omit the square root for performance
 */
func euclideanDistance(a []float32, b []float32) (dist float32) {
	for i := range a {
		diff := a[i] - b[i]
		dist += diff * diff
	}
	return
}

type neighbor struct {
	id       int64
	distance float32
}

/**
* SortedNeighbors maintains a sorted list of the k closest neighbors found so far.
* Neighbors are sorted by distance in ascending order (closest first).
 */
type sortedNeighbors []neighbor

func (h sortedNeighbors) InsertSorted(n neighbor, k int) sortedNeighbors {
	for i := range len(h) {
		if n.distance < h[i].distance {
			h = append(h[:i+1], h[i:]...)
			h[i] = n
			if len(h) > k {
				h = h[:k]
			}
			return h
		}
	}
	// Append if distance is larger than all existing but list is under capacity
	if len(h) < k {
		h = append(h, n)
	}
	return h
}

// nearestNeighborsSequential performs brute-force k-NN search sequentially (used for small datasets).
func nearestNeighborsSequential(query Vector, rawData []DataRow, k int) sortedNeighbors {
	sorted := make(sortedNeighbors, 0, k)
	for _, row := range rawData {
		dist := euclideanDistance(query, row.Vector)
		sorted = sorted.InsertSorted(neighbor{id: row.Id, distance: dist}, k)
	}
	return sorted
}

// mergeNeighbors merges multiple sorted neighbor lists into a single sorted list of k nearest.
func mergeNeighbors(lists []sortedNeighbors, k int) sortedNeighbors {
	merged := make(sortedNeighbors, 0, k)
	for _, list := range lists {
		for _, n := range list {
			merged = merged.InsertSorted(n, k)
		}
	}
	return merged
}

// nearestNeighbors performs parallel brute-force k-NN search to find true nearest neighbors.
func nearestNeighbors(query Vector, rawData []DataRow, k int) []int64 {
	numWorkers := runtime.NumCPU()
	dataLen := len(rawData)

	// Split data into chunks for parallel processing
	chunkSize := (dataLen + numWorkers - 1) / numWorkers
	results := make([]sortedNeighbors, numWorkers)
	var wg sync.WaitGroup

	for i := range numWorkers {
		start := i * chunkSize
		if start >= dataLen {
			break
		}
		end := min(start+chunkSize, dataLen)

		wg.Add(1)
		go func(workerIdx int, chunk []DataRow) {
			defer wg.Done()
			results[workerIdx] = nearestNeighborsSequential(query, chunk, k)
		}(i, rawData[start:end])
	}

	wg.Wait()

	// Merge results from all workers
	merged := mergeNeighbors(results, k)

	resultIds := make([]int64, len(merged))
	for i := range merged {
		resultIds[i] = merged[i].id
	}
	return resultIds
}

func calculateRecall(queryVector Vector, resultIds []int64, rawData []DataRow) float64 {
	// Avoid divide by zero
	if len(resultIds) == 0 {
		return 0.0
	}

	trueNeighbors := nearestNeighbors(queryVector, rawData, len(resultIds))
	trueNeighborMap := make(map[int64]bool)
	for _, id := range trueNeighbors {
		trueNeighborMap[id] = true
	}

	matches := 0
	for _, id := range resultIds {
		if trueNeighborMap[id] {
			matches++
		}
	}

	return float64(matches) / float64(len(resultIds))
}

// EnhanceJobResults calculates recall for all jobs concurrently and returns enhanced results.
func EnhanceJobResults(rawData []DataRow, jobs []Job) []EnhancedJobResult {
	numJobs := len(jobs)
	enhancedResults := make([]EnhancedJobResult, numJobs)

	// Use a worker pool to process jobs concurrently (based on number of CPU cores)
	numWorkers := min(runtime.NumCPU(), numJobs)
	jobChan := make(chan int, numJobs)
	var wg sync.WaitGroup

	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobChan {
				job := jobs[idx]
				recall := calculateRecall(job.QueryVector, job.ResultIds, rawData)
				enhancedResults[idx] = EnhancedJobResult{Job: job, Recall: recall}
			}
		}()
	}

	// Send jobs to workers
	for i := range jobs {
		jobChan <- i
	}
	close(jobChan)

	wg.Wait()
	return enhancedResults
}
